# Self-Confidence vs Other-Confidence: Orthogonalization Analysis

## Motivation

This analysis examines the relationship between the model's self-confidence predictions and other-confidence predictions by orthogonalizing the direction vectors derived from each task.

### The Problem

We have observed evidence for apparent self-knowledge in the confidence task:
- The model's stated confidence correlates with its actual MC uncertainty (entropy, logit gap)
- Uncertainty directions trained on MC task activations transfer to meta-task activations
- Steering and ablation with these directions causally affect stated confidence

However, self-confidence and other-confidence predictions may share common components (e.g., surface difficulty cues). Orthogonalization helps separate unique components.

### The Control Task

We have two meta-tasks:
- **Self-confidence**: "How confident are you that you know the correct answer to this question?"
- **Other-confidence**: "What percentage of college-educated people know the correct answer to this question?"

Both tasks correlate with MC uncertainty and with each other. The vectors derived from each task are similar (shared component) but also different (unique components).

## Approach: Orthogonalization

### Mathematical Framework

Given:
- `d_self_confidence`: Direction trained on self-confidence
- `d_other_confidence`: Direction trained on other-confidence

We compute orthogonalized residuals via Gram-Schmidt projection:

```
d_self_confidence_unique = d_self_confidence - (d_self_confidence · d_other_confidence) * d_other_confidence
d_other_confidence_unique = d_other_confidence - (d_other_confidence · d_self_confidence) * d_self_confidence
```

After normalization:
- `d_self_confidence_unique`: Component of d_self_confidence that is orthogonal to d_other_confidence
- `d_other_confidence_unique`: Component of d_other_confidence that is orthogonal to d_self_confidence

Note: `||d_self_confidence_unique|| = ||d_other_confidence_unique|| = sqrt(1 - cos²(d_self_confidence, d_other_confidence))` by symmetry.

### What We Test

**Predictive Power (4x2 matrix)**:
| Direction | Self-Task | Other-Task |
|-----------|-----------|------------|
| d_self_confidence | Baseline | Cross-task |
| d_other_confidence | Cross-task | Baseline |
| d_self_confidence_unique | Key test | Should be ~0 |
| d_other_confidence_unique | Should be ~0 | Key test |

Key predictions:
- d_self_confidence_unique should still predict self-confidence (R² > 0)
- d_self_confidence_unique should NOT predict other-confidence (orthogonal by construction)

**Causal Experiments (Steering & Ablation)**:

Steering: Does adding the direction shift confidence in the expected direction?
Ablation: Does removing the direction component degrade conf-uncertainty correlation?

Key predictions:
- If d_self_confidence_unique retains causal efficacy on self-task: the unique component is meaningful
- If d_self_confidence_unique has no effect: self-confidence is mostly shared with other-confidence
- If d_other_confidence_unique affects self-task: shared cues contribute to self-confidence too

## Interpretation Guide

### Scenario 1: Strong Unique Components
- cos(d_self_confidence, d_other_confidence) is moderate (~0.3-0.7): directions are related but distinct
- d_self_confidence_unique predicts self-confidence with reasonable R²
- Steering/ablation with d_self_confidence_unique affects self-task but not other-task
- d_other_confidence_unique has little effect on self-task

### Scenario 2: Self-Confidence is Mostly Shared
- cos(d_self_confidence, d_other_confidence) is very high (~0.9+): directions are nearly identical
- d_self_confidence_unique has near-zero predictive power
- Steering/ablation with d_self_confidence_unique has no significant effect
- d_other_confidence predicts self-confidence as well as d_self_confidence

### Scenario 3: Mixed Evidence
- Moderate cosine similarity
- d_self_confidence_unique has some predictive power but less than d_self_confidence
- Causal effects are weaker for d_self_confidence_unique than d_self_confidence
- Suggests unique component contributes but shared signal dominates

## Implementation Details

### Script: `analyze_introspection_orthogonalization.py`

**Prerequisites**:
1. Run `test_meta_transfer.py` with `META_TASK="confidence"` and `FIND_CONFIDENCE_DIRECTIONS=True`
2. Run `test_meta_transfer.py` with `META_TASK="other_confidence"` and `FIND_CONFIDENCE_DIRECTIONS=True`

These generate:
- `*_meta_confidence_metaconfdir_directions.npz`
- `*_meta_other_confidence_metaconfdir_directions.npz`
- Cached activations for both tasks

**Key Functions**:

1. `orthogonalize_directions()`: Gram-Schmidt projection with degenerate detection
2. `compute_predictive_power_matrix()`: R² for all (direction, task) pairs
3. `run_steering_matrix()`: Steering experiments with pooled null + significance
4. `run_ablation_matrix()`: Ablation experiments measuring Δcorrelation

**Degenerate Handling**:
- Layers where cos(d_self_confidence, d_other_confidence) > 0.9 (or residual_norm < 0.1) are flagged
- These layers excluded from aggregate statistics
- Still computed for completeness but interpretation is limited

**Statistical Framework**:
- Predictive: R² with bootstrap percentile CI
- Steering: Slope with pooled null from control directions
- Ablation: Δcorrelation with pooled null from control directions
- P-values: Two-tailed tests against pooled null distribution

### Outputs

1. `*_orthogonal_directions.npz`: d_self_confidence_unique + d_other_confidence_unique vectors per layer
2. `*_orthogonal_analysis_results.json`: Full results including:
   - Orthogonalization stats (cosine, residual_norm, shared/unique variance)
   - Predictive power matrix
   - Steering results
   - Ablation results
   - Interpretation summary
3. Visualization plots:
   - `*_orthogonal_similarity.png`: Cosine similarity by layer
   - `*_orthogonal_predictive.png`: 4x2 predictive power heatmap
   - `*_orthogonal_steering.png`: Steering effects
   - `*_orthogonal_ablation.png`: Ablation effects

## Limitations and Caveats

1. **Orthogonalization is not perfect separation**: Projecting out d_other_confidence removes the component in that direction, but doesn't guarantee perfect separation of concepts. The residual may still contain shared information not captured by d_other_confidence.

2. **Other-confidence is not purely external**: The model might use some self-knowledge even when estimating human performance ("I find this hard, so humans probably do too").

3. **Layer-by-layer analysis**: Directions and their relationships vary by layer. A direction might have different properties at different layers.

4. **Degenerate cases**: When d_self_confidence ≈ d_other_confidence, orthogonalization yields noise. These layers are flagged but still included.

5. **Causal experiments on different tasks**: Running steering/ablation on different tasks (self vs other) involves different prompts and may have different baselines.

## References

- Main confidence analysis: `test_meta_transfer.py`
- Direction finding: `core/confidence_directions.py`
- Steering infrastructure: `run_steering_causality.py`
- Ablation infrastructure: `run_ablation_causality.py`
- Task definitions: `tasks.py` (STATED_CONFIDENCE_SETUP, OTHER_CONFIDENCE_SETUP)
