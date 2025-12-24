"""
Core utilities for introspection experiments.

Modules:
- model_utils: Model loading, naming, device detection
- extraction: BatchedExtractor for activation/logit extraction
- probes: Linear probe training, transfer testing, permutation tests
- questions: Question loading, hashing, consistency verification
- steering: Steering and ablation hooks for activation intervention
"""

from .model_utils import (
    DEVICE,
    is_base_model,
    has_chat_template,
    get_model_short_name,
    get_run_name,
    load_model_and_tokenizer,
    should_use_chat_template,
)

from .extraction import (
    compute_entropy_from_probs,
    BatchedExtractor,
    extract_activations_only,
)

from .probes import (
    LinearProbe,
    train_and_evaluate_probe,
    test_transfer,
    permutation_test,
    run_layer_analysis,
    # Introspection mapping
    compute_introspection_scores,
    train_introspection_mapping_probe,
    compute_contrastive_direction,
    extract_probe_direction,
    run_introspection_mapping_analysis,
)

from .questions import (
    load_questions,
    get_question_hash,
    save_question_set,
    load_question_set,
    verify_question_consistency,
    format_direct_prompt,
    split_questions,
)

from .steering import (
    SteeringHook,
    AblationHook,
    generate_orthogonal_directions,
    steering_context,
    ablation_context,
    multi_layer_steering_context,
    compute_projection_magnitude,
    measure_steering_effect,
)

__all__ = [
    # model_utils
    "DEVICE",
    "is_base_model",
    "has_chat_template",
    "get_model_short_name",
    "get_run_name",
    "load_model_and_tokenizer",
    "should_use_chat_template",
    # extraction
    "compute_entropy_from_probs",
    "BatchedExtractor",
    "extract_activations_only",
    # probes
    "LinearProbe",
    "train_and_evaluate_probe",
    "test_transfer",
    "permutation_test",
    "run_layer_analysis",
    # introspection mapping
    "compute_introspection_scores",
    "train_introspection_mapping_probe",
    "compute_contrastive_direction",
    "extract_probe_direction",
    "run_introspection_mapping_analysis",
    # questions
    "load_questions",
    "get_question_hash",
    "save_question_set",
    "load_question_set",
    "verify_question_consistency",
    "format_direct_prompt",
    "split_questions",
    # steering
    "SteeringHook",
    "AblationHook",
    "generate_orthogonal_directions",
    "steering_context",
    "ablation_context",
    "multi_layer_steering_context",
    "compute_projection_magnitude",
    "measure_steering_effect",
]
