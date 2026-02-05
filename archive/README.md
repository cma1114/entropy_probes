# Archive

Scripts in this directory are **superseded** by the current modular pipeline. They are kept for reference and historical context but are not part of the active workflow.

## Legacy scripts (superseded by current workflow)

| Script | Superseded by |
|--------|--------------|
| `mc_entropy_probe.py` | `identify_mc_correlate.py` |
| `nexttoken_entropy_probe.py` | `identify_nexttoken_correlate.py` |
| `run_introspection_experiment.py` | `test_meta_transfer.py` |
| `run_introspection_steering.py` | `run_steering_causality.py` |
| `run_contrastive_direction.py` | `run_ablation_causality.py` + `run_steering_causality.py` |
| `run_introspection_probe.py` | `identify_confidence_correlate.py` (merged into `test_meta_transfer.py`) |
| `run_introspection_direction_experiment.py` | Modular pipeline |
| `run_mc_answer_ablation.py` | `run_ablation_causality.py` with `DIRECTION_TYPE="answer"` |
| `test_cross_dataset_transfer_patched.py` | Fixes folded into `test_cross_dataset_transfer.py` |

## One-off analysis scripts

| Script | Purpose |
|--------|---------|
| `analyze_shared_unique.py` | Shared vs dataset-specific direction decomposition |
| `analyze_direction_mixture.py` | Cross-layer direction mixture analysis |
| `analyze_mc_answer_bias.py` | Answer position bias analysis |
| `compare_cross_dataset_directions.py` | Permutation-based cross-dataset direction comparison |
| `compute_contrastive_directions.py` | Contrastive direction computation (old approach) |
| `visualize_contrastive_ablation.py` | Visualization for contrastive ablation results |
| `generate_report.py` | Hardcoded report generator |
| `run_ablation_swap_directions.py` | Direction swapping experiment |

## Debug/diagnostic utilities

| Script | Purpose |
|--------|---------|
| `debug_activation_stats.py` | Activation statistics debugging |
| `diagnose_direction_overlap.py` | Direction overlap diagnosis |
| `verify_baseline_correlation.py` | Baseline correlation verification |
| `regenerate_position_plots.py` | Plot regeneration utility |

## External utilities

| Script | Purpose |
|--------|---------|
| `logres_helpers.py` | Logistic regression helper functions |
