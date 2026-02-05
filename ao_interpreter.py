"""
Interpret linear probe directions using Activation Oracles.

Usage:
    from interpret_probe_with_ao import ProbeInterpreter
    
    interpreter = ProbeInterpreter(model_size="8b")  # or "70b"
    
    # Your function that returns probe vectors of shape (num_layers, d_resid)
    probe_vectors = get_my_probe_vectors()
    
    # Interpret a specific layer's probe direction
    result = interpreter.interpret(
        vector=probe_vectors[40],  # e.g., layer 40
        source_layer=40,
        question="What concept does this direction represent?"
    )
    print(result)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional, Callable, List
import functools


# Model configurations
MODEL_CONFIGS = {
    "8b": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "ao_adapter": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
        "num_layers": 32,
        "d_model": 4096,
    },
    "70b": {
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "ao_adapter": "adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct",
        "num_layers": 80,
        "d_model": 8192,
    },
}

PLACEHOLDER_TOKEN = "?"


class ProbeInterpreter:
    def __init__(
        self,
        model_size: str = "8b",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,  # Useful for 70B
    ):
        """
        Initialize the Activation Oracle for interpreting probe directions.
        
        Args:
            model_size: "8b" or "70b"
            device: Device to load model on
            torch_dtype: Model precision
            load_in_8bit: Whether to use 8-bit quantization (recommended for 70B)
        """
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")
        
        self.config = MODEL_CONFIGS[model_size]
        self.device = device
        
        print(f"Loading base model: {self.config['base_model']}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"],
            torch_dtype=torch_dtype,
            device_map=device,
            load_in_8bit=load_in_8bit,
        )
        
        print(f"Loading AO adapter: {self.config['ao_adapter']}")
        self.model = PeftModel.from_pretrained(self.model, self.config["ao_adapter"])
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
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
        placeholder_positions: list[int],
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
    
    def _find_placeholder_positions(self, input_ids: torch.Tensor) -> list[int]:
        """Find positions of placeholder tokens in the input."""
        positions = (input_ids[0] == self.placeholder_id).nonzero(as_tuple=True)[0]
        return positions.tolist()
    
    def _get_layer_1_module(self):
        """Get the module after which to inject (layer 1)."""
        # For Llama models, layers are in model.model.layers
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
    
    def interpret_with_confidence(
        self,
        vector: torch.Tensor,
        source_layer: int,
        question: str = "What concept or phenomenon does this direction represent?",
        max_new_tokens: int = 256,
        num_placeholders: int = 1,
    ) -> dict:
        """
        Interpret a direction and return confidence metrics.
        
        Returns:
            dict with keys:
                - response: The generated text
                - mean_logprob: Average log probability of generated tokens
                - min_logprob: Minimum log probability (weakest link)
                - logprobs: Full list of per-token logprobs
                - tokens: The generated tokens
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
        vectors = vector.unsqueeze(0)
        
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
                    do_sample=False,  # Greedy for reproducibility
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Extract generated token ids
            generated_ids = outputs.sequences[0, input_ids.shape[1]:]
            
            # Compute log probabilities for each generated token
            logprobs = []
            tokens = []
            for i, score in enumerate(outputs.scores):
                # score shape: (batch=1, vocab_size)
                probs = torch.softmax(score[0], dim=-1)
                token_id = generated_ids[i].item()
                token_logprob = torch.log(probs[token_id] + 1e-10).item()
                logprobs.append(token_logprob)
                tokens.append(self.tokenizer.decode([token_id]))
            
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        finally:
            handle.remove()
        
        return {
            "response": response.strip(),
            "mean_logprob": sum(logprobs) / len(logprobs) if logprobs else 0,
            "min_logprob": min(logprobs) if logprobs else 0,
            "logprobs": logprobs,
            "tokens": tokens,
        }
    
    def interpret_with_consistency(
        self,
        vector: torch.Tensor,
        source_layer: int,
        question: str = "What concept or phenomenon does this direction represent?",
        num_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ) -> dict:
        """
        Sample multiple interpretations to measure consistency.
        
        Returns:
            dict with keys:
                - responses: List of all sampled responses
                - unique_responses: Number of meaningfully different responses
                - consensus: Most common response (if any agreement)
        """
        responses = []
        for i in range(num_samples):
            response = self.interpret(
                vector=vector,
                source_layer=source_layer,
                question=question,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            responses.append(response)
            print(f"  Sample {i+1}: {response[:80]}...")
        
        return {
            "responses": responses,
            "num_samples": num_samples,
        }
    
    def compare_to_baseline(
        self,
        vector: torch.Tensor,
        source_layer: int,
        question: str = "What concept or phenomenon does this direction represent?",
        num_random_baselines: int = 3,
        **kwargs,
    ) -> dict:
        """
        Compare interpretation of your vector against random baselines.
        
        If the AO says similar things about random vectors, it's probably
        not picking up on anything real in your probe direction.
        
        Returns:
            dict with keys:
                - probe_result: Interpretation of your actual probe
                - baseline_results: List of interpretations for random vectors
                - probe_logprob: Mean logprob for probe interpretation
                - baseline_logprobs: Mean logprobs for baselines
        """
        print("Interpreting probe direction...")
        probe_result = self.interpret_with_confidence(
            vector=vector,
            source_layer=source_layer,
            question=question,
            **kwargs,
        )
        
        baseline_results = []
        baseline_logprobs = []
        
        for i in range(num_random_baselines):
            print(f"Interpreting random baseline {i+1}/{num_random_baselines}...")
            # Random vector with same norm as probe
            random_vec = torch.randn_like(vector)
            random_vec = random_vec / random_vec.norm() * vector.norm()
            
            result = self.interpret_with_confidence(
                vector=random_vec,
                source_layer=source_layer,
                question=question,
                **kwargs,
            )
            baseline_results.append(result["response"])
            baseline_logprobs.append(result["mean_logprob"])
        
        return {
            "probe_result": probe_result["response"],
            "probe_logprob": probe_result["mean_logprob"],
            "probe_min_logprob": probe_result["min_logprob"],
            "baseline_results": baseline_results,
            "baseline_logprobs": baseline_logprobs,
            "baseline_mean_logprob": sum(baseline_logprobs) / len(baseline_logprobs),
        }

    def full_analysis(
        self,
        vector: torch.Tensor,
        source_layer: int,
        questions: Optional[list[str]] = None,
    ) -> dict:
        """
        Run a full analysis: multiple questions, consistency check, baseline comparison.
        """
        if questions is None:
            questions = [
                "What concept does this direction represent?",
                "Is this related to uncertainty or confidence?",
                "What kind of text would strongly activate this direction?",
            ]
        
        results = {"layer": source_layer, "questions": {}}
        
        for question in questions:
            print(f"\n{'='*60}")
            print(f"Q: {question}")
            print('='*60)
            
            # Get interpretation with confidence
            conf_result = self.interpret_with_confidence(
                vector=vector,
                source_layer=source_layer,
                question=question,
            )
            print(f"Response: {conf_result['response']}")
            print(f"Mean logprob: {conf_result['mean_logprob']:.3f}")
            print(f"Min logprob: {conf_result['min_logprob']:.3f}")
            
            results["questions"][question] = conf_result
        
        # Baseline comparison on first question
        print(f"\n{'='*60}")
        print("Comparing to random baselines...")
        print('='*60)
        baseline_result = self.compare_to_baseline(
            vector=vector,
            source_layer=source_layer,
            question=questions[0],
        )
        results["baseline_comparison"] = baseline_result
        
        print(f"\nProbe logprob: {baseline_result['probe_logprob']:.3f}")
        print(f"Baseline mean logprob: {baseline_result['baseline_mean_logprob']:.3f}")
        print(f"Delta: {baseline_result['probe_logprob'] - baseline_result['baseline_mean_logprob']:.3f}")
        
        print("\nBaseline responses:")
        for i, resp in enumerate(baseline_result["baseline_results"]):
            print(f"  Random {i+1}: {resp[:100]}...")
        
        return results

    def interpret_multiple_layers(
        self,
        vectors: torch.Tensor,
        layers: Optional[list[int]] = None,
        question: str = "What concept or phenomenon does this direction represent?",
        **kwargs,
    ) -> dict[int, str]:
        """
        Interpret probe directions from multiple layers.
        
        Args:
            vectors: Probe directions, shape (num_layers, d_model)
            layers: Which layers to interpret (default: middle layers at 25%, 50%, 75% depth)
            question: Question to ask
            **kwargs: Additional arguments passed to interpret()
        
        Returns:
            Dictionary mapping layer index to interpretation
        """
        num_layers = vectors.shape[0]
        
        if layers is None:
            # Default to 25%, 50%, 75% depth (what AO was trained on)
            layers = [
                int(num_layers * 0.25),
                int(num_layers * 0.50),
                int(num_layers * 0.75),
            ]
        
        results = {}
        for layer in layers:
            print(f"Interpreting layer {layer}...")
            results[layer] = self.interpret(
                vector=vectors[layer],
                source_layer=layer,
                question=question,
                **kwargs,
            )
        
        return results
    
    def compare_directions(
        self,
        vector1: torch.Tensor,
        vector2: torch.Tensor,
        source_layer: int,
        label1: str = "Direction A",
        label2: str = "Direction B",
    ) -> dict[str, str]:
        """
        Compare two directions (e.g., entropy probe vs. random direction).
        
        Returns interpretations for both and their difference.
        """
        results = {}
        
        results[label1] = self.interpret(
            vector=vector1,
            source_layer=source_layer,
            question="What concept does this direction represent?",
        )
        
        results[label2] = self.interpret(
            vector=vector2,
            source_layer=source_layer,
            question="What concept does this direction represent?",
        )
        
        # Also try the difference
        diff = vector1 - vector2
        results[f"{label1} - {label2}"] = self.interpret(
            vector=diff,
            source_layer=source_layer,
            question="What is the difference between these two concepts?",
        )
        
        return results


# =============================================================================
# Example usage and questions to try
# =============================================================================

SUGGESTED_QUESTIONS = [
    "What concept or phenomenon does this direction represent?",
    "What is the model thinking about when this direction is active?",
    "Is this related to uncertainty or confidence?",
    "Does this direction relate to the model's certainty about its next prediction?",
    "What kind of text would activate this direction strongly?",
    "Is this direction associated with any particular type of reasoning?",
    "Describe the semantic content of this direction.",
]


def example_usage():
    """Example showing how to use the interpreter."""
    
    # Placeholder for your probe vectors
    def get_my_probe_vectors() -> torch.Tensor:
        """
        Replace this with your actual probe loading code.
        Should return tensor of shape (num_layers, d_model).
        """
        # Example for 8B model:
        num_layers, d_model = 32, 4096
        # In reality, load your trained probe weights here
        return torch.randn(num_layers, d_model)
    
    # Initialize interpreter
    interpreter = ProbeInterpreter(model_size="8b")
    
    # Get your probe vectors
    probe_vectors = get_my_probe_vectors()
    
    # Pick a layer where your probe works well
    layer = 20
    
    # ==========================================================================
    # Option 1: Simple interpretation (no confidence info)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SIMPLE INTERPRETATION")
    print('='*60)
    
    response = interpreter.interpret(
        vector=probe_vectors[layer],
        source_layer=layer,
        question="What concept does this direction represent?",
    )
    print(f"Response: {response}")
    
    # ==========================================================================
    # Option 2: Interpretation with confidence (log probabilities)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("INTERPRETATION WITH CONFIDENCE")
    print('='*60)
    
    result = interpreter.interpret_with_confidence(
        vector=probe_vectors[layer],
        source_layer=layer,
        question="What concept does this direction represent?",
    )
    print(f"Response: {result['response']}")
    print(f"Mean logprob: {result['mean_logprob']:.3f}")
    print(f"Min logprob: {result['min_logprob']:.3f}")
    
    # Higher (less negative) = more confident
    # Typical range: -0.5 (very confident) to -3.0 (uncertain)
    
    # ==========================================================================
    # Option 3: Compare against random baselines
    # ==========================================================================
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print('='*60)
    
    comparison = interpreter.compare_to_baseline(
        vector=probe_vectors[layer],
        source_layer=layer,
        question="What concept does this direction represent?",
        num_random_baselines=3,
    )
    
    print(f"\nYour probe: {comparison['probe_result']}")
    print(f"Your probe logprob: {comparison['probe_logprob']:.3f}")
    print(f"\nRandom baselines:")
    for i, (resp, lp) in enumerate(zip(
        comparison['baseline_results'], 
        comparison['baseline_logprobs']
    )):
        print(f"  {i+1}. (logprob={lp:.3f}) {resp[:80]}...")
    
    # If probe_logprob >> baseline_logprobs, the AO is more confident
    # about your probe than random noise. Good sign!
    
    # ==========================================================================
    # Option 4: Full analysis (recommended for serious investigation)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("FULL ANALYSIS")
    print('='*60)
    
    results = interpreter.full_analysis(
        vector=probe_vectors[layer],
        source_layer=layer,
    )


if __name__ == "__main__":
    example_usage()