# Introspection vs Surface Difficulty: Orthogonalization Analysis

## Motivation

This analysis addresses a fundamental question about the nature of the model's confidence representations: **Is the model genuinely introspecting on its own uncertainty, or merely recognizing surface difficulty cues in the data?**

### The Problem

We have observed remarkably good evidence for apparent introspection in the confidence task:
- The model's stated confidence correlates strongly with its actual MC uncertainty (entropy, logit gap)
- Uncertainty directions trained on MC task activations transfer well to meta-task activations
- Steering and ablation with these directions causally affect stated confidence

However, a skeptical interpretation exists: the model might be picking up **surface cues of difficulty** in the questions themselves (e.g., obscure topics, complex phrasing, ambiguous wording) rather than truly introspecting on its internal uncertainty state.

### The Control Task

To address this, we have a second meta-task called "other-confidence":
- **Self-confidence**: "How confident are you that you know the correct answer to this question?"
- **Other-confidence**: "What percentage of college-educated people know the correct answer to this question?"

The key insight is:
- **Self-confidence** should capture: true introspection + surface difficulty cues
- **Other-confidence** should capture: primarily surface difficulty cues (model estimating human difficulty)

Both tasks will show some correlation with MC uncertainty, and both will correlate with each other. The vectors derived from each task will be somewhat similar (due to shared surface cue component) but also somewhat different (due to unique introspection component in self-confidence).

## Approach: Orthogonalization

### Mathematical Framework

Given:
- `d_self`: Direction trained on self-confidence (captures introspection + surface)
- `d_other`: Direction trained on other-confidence (captures primarily surface)

We compute orthogonalized residuals via Gram-Schmidt projection:

```
d_introspection = d_self - (d_self · d_other) * d_other
d_surface = d_other - (d_other · d_self) * d_self
```

After normalization:
- `d_introspection`: Component of d_self that is orthogonal to d_other (pure introspection)
- `d_surface`: Component of d_other that is orthogonal to d_self (pure surface cues)

Note: `||d_introspection|| = ||d_surface|| = sqrt(1 - cos²(d_self, d_other))` by symmetry.

### What We Test

**Predictive Power (4x2 matrix)**:
| Direction | Self-Task | Other-Task |
|-----------|-----------|------------|
| d_self | Baseline | Cross-task |
| d_other | Cross-task | Baseline |
| d_introspection | Key test | Should be ~0 |
| d_surface | Should be ~0 | Key test |

If true introspection exists:
- d_introspection should still predict self-confidence (R² > 0)
- d_introspection should NOT predict other-confidence (orthogonal by construction)

**Causal Experiments (Steering & Ablation)**:

Steering: Does adding the direction shift confidence in the expected direction?
Ablation: Does removing the direction component degrade conf-uncertainty correlation?

Key predictions:
- If d_introspection retains causal efficacy on self-task: true introspection exists
- If d_introspection has no effect: self-confidence is just surface difficulty estimation
- If d_surface affects self-task: surface cues contribute to self-confidence too

## Interpretation Guide

### Scenario 1: Strong Evidence for True Introspection
- cos(d_self, d_other) is moderate (~0.3-0.7): directions are related but distinct
- d_introspection predicts self-confidence with reasonable R²
- Steering/ablation with d_introspection affects self-task but not other-task
- d_surface has little effect on self-task

### Scenario 2: Self-Confidence is Mostly Surface Difficulty
- cos(d_self, d_other) is very high (~0.9+): directions are nearly identical
- d_introspection has near-zero predictive power
- Steering/ablation with d_introspection has no significant effect
- d_other predicts self-confidence as well as d_self

### Scenario 3: Mixed Evidence
- Moderate cosine similarity
- d_introspection has some predictive power but less than d_self
- Causal effects are weaker for d_introspection than d_self
- Suggests introspection contributes but surface cues dominate

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
- Layers where cos(d_self, d_other) > 0.9 (or residual_norm < 0.1) are flagged
- These layers excluded from aggregate statistics
- Still computed for completeness but interpretation is limited

**Statistical Framework**:
- Predictive: R² with bootstrap percentile CI
- Steering: Slope with pooled null from control directions
- Ablation: Δcorrelation with pooled null from control directions
- P-values: Two-tailed tests against pooled null distribution

### Outputs

1. `*_orthogonal_directions.npz`: d_introspection + d_surface vectors per layer
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

1. **Orthogonalization is not perfect separation**: Projecting out d_other removes the component in that direction, but doesn't guarantee perfect separation of concepts. The residual may still contain some surface cue information not captured by d_other.

2. **Other-confidence is not pure surface cues**: The model might use some introspection even when estimating human performance ("I find this hard, so humans probably do too").

3. **Layer-by-layer analysis**: Directions and their relationships vary by layer. A direction might be introspective at one layer but not another.

4. **Degenerate cases**: When d_self ≈ d_other, orthogonalization yields noise. These layers are flagged but still included.

5. **Causal experiments on different tasks**: Running steering/ablation on different tasks (self vs other) involves different prompts and may have different baselines.

## References

- Main confidence analysis: `test_meta_transfer.py`
- Direction finding: `core/confidence_directions.py`
- Steering infrastructure: `run_steering_causality.py`
- Ablation infrastructure: `run_ablation_causality.py`
- Task definitions: `tasks.py` (STATED_CONFIDENCE_SETUP, OTHER_CONFIDENCE_SETUP)
