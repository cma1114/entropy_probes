"""
Activation and logit extraction utilities.

Provides BatchedExtractor for efficient combined extraction of:
- Layer activations (at last token position)
- Logits over specified option tokens
- Entropy computation
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute entropy from a probability distribution."""
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy)


class BatchedExtractor:
    """
    Extract activations and logits in a single batched forward pass.

    This class registers forward hooks on all model layers to capture
    hidden states, then extracts both activations and option probabilities
    in one forward pass (instead of separate passes).
    """

    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.activations = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures hidden states for a layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach()
        return hook

    def register_hooks(self):
        """Register forward hooks on all model layers."""
        if hasattr(self.model, 'get_base_model'):
            base = self.model.get_base_model()
            layers = base.model.layers
        else:
            layers = self.model.model.layers

        for i, layer in enumerate(layers):
            hook = self._make_hook(i)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_token_ids: List[int]
    ) -> Tuple[List[Dict[int, np.ndarray]], List[np.ndarray], List[float]]:
        """
        Extract activations AND compute option probabilities in one forward pass.

        Args:
            input_ids: (batch_size, seq_len) tensor of input token IDs
            attention_mask: (batch_size, seq_len) attention mask
            option_token_ids: List of token IDs for the answer options

        Returns:
            layer_activations: List of {layer_idx: activation} dicts, one per batch item
            option_probs: List of probability arrays over options, one per batch item
            entropies: List of entropy values, one per batch item
        """
        self.activations = {}
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Get last token index for each item in batch
        seq_lengths = attention_mask.sum(dim=1)  # (batch_size,)

        # Extract activations at last token for each batch item
        all_layer_activations = []
        for batch_idx in range(batch_size):
            last_idx = seq_lengths[batch_idx] - 1
            item_activations = {}
            for layer_idx, acts in self.activations.items():
                item_activations[layer_idx] = acts[batch_idx, last_idx, :].cpu().numpy()
            all_layer_activations.append(item_activations)

        # Extract logits and compute probabilities for each batch item
        all_probs = []
        all_entropies = []
        for batch_idx in range(batch_size):
            last_idx = seq_lengths[batch_idx] - 1
            final_logits = outputs.logits[batch_idx, last_idx, :]
            option_logits = final_logits[option_token_ids]
            probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
            entropy = compute_entropy_from_probs(probs)
            all_probs.append(probs)
            all_entropies.append(entropy)

        return all_layer_activations, all_probs, all_entropies

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False


def extract_activations_only(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_layers: int
) -> Dict[int, np.ndarray]:
    """
    Extract activations from all layers for a single input (no option probs).

    Simpler interface when you just need activations.

    Args:
        model: The transformer model
        input_ids: (1, seq_len) or (seq_len,) tensor
        attention_mask: Matching attention mask
        num_layers: Number of layers in model

    Returns:
        Dict mapping layer_idx to activation array of shape (hidden_dim,)
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            activations[layer_idx] = hidden_states.detach()
        return hook

    # Get layers
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
        layers = base.model.layers
    else:
        layers = model.model.layers

    # Register hooks
    for i, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook(i))
        hooks.append(handle)

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Extract at last token
        seq_len = attention_mask.sum().item()
        last_idx = seq_len - 1

        result = {}
        for layer_idx, acts in activations.items():
            result[layer_idx] = acts[0, last_idx, :].cpu().numpy()

        return result

    finally:
        for handle in hooks:
            handle.remove()
