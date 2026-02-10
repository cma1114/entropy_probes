"""
Core utilities for introspection experiments.

Modules:
- model_utils: Model loading, naming, device detection
- extraction: BatchedExtractor for activation/logit extraction
- metrics: Uncertainty metric computation (entropy, logit_gap, etc.)
- directions: Direction finding methods (probe, mean_diff)
- answer_directions: MC answer choice direction finding (4-class classification)
- confidence_directions: Meta-judgment confidence direction finding
- probes: Linear probe training, transfer testing, permutation tests
- questions: Question loading, hashing, consistency verification
- steering: Steering and ablation hooks for activation intervention
- steering_experiments: Experiment runners and statistical analysis
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
)

from .metrics import (
    compute_entropy,
    compute_metrics_single,
    compute_mc_metrics,
    compute_nexttoken_metrics,
    METRIC_INFO,
    metric_sign_for_confidence,
)

from .directions import (
    probe_direction,
    mean_diff_direction,
    find_directions,
    apply_direction,
    apply_probe_shared,
    apply_probe_centered,
    apply_probe_separate,
    evaluate_transfer,
)

from .answer_directions import (
    train_mc_answer_classifier,
    extract_answer_direction,
    apply_answer_classifier_centered,
    apply_answer_classifier_separate,
    find_answer_directions,
    find_answer_directions_both_methods,
    class_centroid_direction,
    encode_answers,
    decode_answers,
)

from .confidence_directions import (
    train_confidence_probe,
    extract_confidence_direction,
    evaluate_confidence_probe,
    find_confidence_directions,
    find_confidence_directions_both_methods,
    find_mc_uncertainty_directions_from_meta,
    compare_confidence_to_uncertainty,
    cross_evaluate_directions,
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

from .steering_experiments import (
    SteeringExperimentConfig,
    # KV cache utilities
    extract_cache_tensors,
    create_fresh_cache,
    get_kv_cache,
    # Batch hooks
    BatchSteeringHook,
    BatchAblationHook,
    # Tokenization
    pretokenize_prompts,
    build_padded_gpu_batches,
    # Direction prep
    precompute_direction_tensors,
    # Experiment runners
    run_steering_experiment,
    run_ablation_experiment,
    # Analysis
    compute_correlation,
    get_expected_slope_sign,
    analyze_steering_results,
    analyze_ablation_results,
    # Printing
    print_steering_summary,
    print_ablation_summary,
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
    # metrics
    "compute_entropy",
    "compute_metrics_single",
    "compute_mc_metrics",
    "compute_nexttoken_metrics",
    "METRIC_INFO",
    "metric_sign_for_confidence",
    # directions
    "probe_direction",
    "mean_diff_direction",
    "find_directions",
    "apply_direction",
    "apply_probe_shared",
    "apply_probe_centered",
    "apply_probe_separate",
    "evaluate_transfer",
    # answer_directions
    "train_mc_answer_classifier",
    "extract_answer_direction",
    "apply_answer_classifier_centered",
    "apply_answer_classifier_separate",
    "find_answer_directions",
    "find_answer_directions_both_methods",
    "class_centroid_direction",
    "encode_answers",
    "decode_answers",
    # confidence_directions
    "train_confidence_probe",
    "extract_confidence_direction",
    "evaluate_confidence_probe",
    "find_confidence_directions",
    "find_confidence_directions_both_methods",
    "find_mc_uncertainty_directions_from_meta",
    "compare_confidence_to_uncertainty",
    "cross_evaluate_directions",
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
    # steering_experiments
    "SteeringExperimentConfig",
    "extract_cache_tensors",
    "create_fresh_cache",
    "get_kv_cache",
    "BatchSteeringHook",
    "BatchAblationHook",
    "pretokenize_prompts",
    "build_padded_gpu_batches",
    "precompute_direction_tensors",
    "run_steering_experiment",
    "run_ablation_experiment",
    "compute_correlation",
    "get_expected_slope_sign",
    "analyze_steering_results",
    "analyze_ablation_results",
    "print_steering_summary",
    "print_ablation_summary",
]
