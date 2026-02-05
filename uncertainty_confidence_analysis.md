# Analysis: Why Ablating Uncertainty Increases Confidence Projection

**Date**: 2024-02-04
**Model**: Llama-3.1-8B-Instruct
**Dataset**: TriviaMC (difficulty filtered)

## The Puzzle

We trained two directions:
1. **Uncertainty direction**: Trained on MC-task logit_gap (difference between top-2 answer logits). High projection = model is certain about MC answer.
2. **Confidence direction**: Trained on meta-task stated confidence (0-100 scale). High projection = model states high confidence.

These directions have:
- **Cosine similarity ~0.5**: The directions share ~50% structural overlap in activation space
- **Projection correlation ~0.96**: For each sample, projections onto the two directions are highly correlated—the directions rank samples almost identically
- **Behavioral correlation ~0.62**: The actual correlation between MC logit_gap and meta-task stated confidence (baseline calibration)

The gap between projection correlation (0.96) and behavioral correlation (0.62) indicates that while the directions capture nearly identical variance in activations, neither direction perfectly predicts its training target. The directions may be capturing a shared underlying signal that's related to—but not identical to—both logit_gap and stated confidence.

When we **ablate the uncertainty direction** at layer L14, something counterintuitive happens: the projection of downstream activations onto the **confidence direction increases** rather than decreases.

## The Explanation: Negative Projection Mechanics

### The Key Finding

**Meta-task activations have NEGATIVE mean projection onto the uncertainty direction (-0.4127 at L14).**

This requires careful interpretation. The uncertainty direction is trained on **MC-task** activations:

| Context | Layer | Mean Projection | Std | Interpretation |
|---------|-------|-----------------|-----|----------------|
| MC-task | L14 | -0.09 | 0.95 | Centered near zero; samples span both sides |
| Meta-task | L14 | -0.41 | 0.11 (SEM) | Significantly negative (z ≈ -3.7) |

During the MC task, samples naturally span both sides of the direction (some certain, some uncertain). But during the **meta-task** (asking "how confident are you?"), activations systematically sit on the negative side.

This makes sense: the meta-task involves introspection/reflection, which puts the model in a different region of activation space than confident MC answering. The MC-uncertainty direction, trained to separate certain from uncertain **MC answers**, sees meta-task activations as "uncertain-ish" because introspection is not the same cognitive state as confidently selecting an answer.

### How Ablation Works

Ablation removes the component of an activation along a direction:
```
x' = x - (x·u)·u
```

When the projection `x·u` is negative:
```
x' = x - (negative)·u = x + |x·u|·u
```

**Ablation with negative projections effectively ADDS the direction rather than removing it.**

### Why This Increases Confidence Projection

1. The uncertainty direction points toward "certainty" (trained on logit_gap, where high = certain)
2. Meta-task activations have negative mean projection (-0.41) onto this direction
3. Ablation pushes these activations toward the "certain" side (toward zero and beyond)
4. The confidence direction is positively aligned with uncertainty (cos ~0.5)
5. Therefore, confidence projection increases

This is not inhibition or suppression—it's vector arithmetic operating on activations that happen to sit on the negative side of the direction.

## Per-Sample Correlation Analysis

To verify this explanation, we computed per-sample correlations. For each sample, we measured:
- **Baseline uncertainty projection**: How far the sample sits along the uncertainty direction before ablation
- **Delta confidence projection**: How much the confidence projection changes after ablation

**What correlation (r) means here**: Pearson correlation between these two quantities across all samples. A negative r means samples with more negative uncertainty projection show larger increases in confidence projection—exactly what the negative projection mechanics predicts.

### Results: L14-L16 Ablations CONFIRM the Explanation

| Ablation Layer | Measure Layer | r | 95% CI | Interpretation |
|----------------|---------------|-----|--------|----------------|
| L14 | L15 | -0.63 | [-0.69, -0.56] | Confirmed |
| L14 | L16 | -0.75 | [-0.80, -0.69] | Confirmed |
| L14 | L17 | -0.84 | [-0.88, -0.80] | Confirmed |
| L15 | L16 | -0.92 | [-0.94, -0.90] | Confirmed |
| L15 | L17 | -0.96 | [-0.97, -0.95] | Confirmed |
| L16 | L17 | -0.96 | [-0.97, -0.95] | Confirmed |

The strong negative correlations confirm: samples with more negative baseline uncertainty projection show larger confidence increases after ablation. This is the predicted signature of negative projection mechanics.

### Anomaly: L17 Ablations Show Opposite Pattern

| Ablation Layer | Measure Layer | r | 95% CI | Interpretation |
|----------------|---------------|-----|--------|----------------|
| L17 | L18 | +0.53 | [+0.43, +0.62] | Unexpected |
| L17 | L19 | +0.66 | [+0.58, +0.73] | Unexpected |
| L17 | L20 | +0.70 | [+0.62, +0.76] | Unexpected |

Something qualitatively different happens at L17. The simple negative projection explanation breaks down here—possibly indicating a transition in how the model processes uncertainty/confidence signals.

### Slope Ratios Indicate Nonlinear Amplification

The ratio of actual slope to theoretical slope (what we'd expect if the effect were purely linear) grows with layer distance:

| Layer Gap | Slope Ratio |
|-----------|-------------|
| 1 layer | 0.16 - 0.23 |
| 2 layers | 0.30 - 0.94 |
| 3 layers | 1.75 - 7.27 |

This suggests the effect compounds through layers rather than simply propagating linearly.

## Why Ablation Breaks Calibration

The per-sample analysis reveals why ablating uncertainty **increases mean confidence** but **degrades calibration**:

### Before Ablation
- Meta-task activations vary along the uncertainty direction (some more negative, some less negative)
- Even though the mean is negative (-0.41), there's variance across samples
- Samples where the model was more certain on the MC task project less negatively
- Samples where the model was less certain project more negatively
- This variance is the signal the model uses to express calibrated confidence

### After Ablation
- All samples have zero component along the uncertainty direction
- The per-sample differentiation is destroyed—all samples are now equidistant from "certainty" along this direction
- Mean confidence increases (negative projections pushed toward zero/certainty)
- But calibration breaks: the model can no longer distinguish which samples it was actually uncertain about

### The Evidence

| Condition | Calibration (r) | Mean Confidence |
|-----------|-----------------|-----------------|
| Baseline | 0.624 | Lower |
| After L15 ablation | 0.519 | Higher |

Calibration = correlation between MC logit_gap and meta-task stated confidence. A drop from 0.62 to 0.52 means the model's stated confidence no longer tracks its actual uncertainty as well.

## Steering Results Confirm Direction Semantics

| Layer | Slope (Δconf/Δmult) | Z-score | Interpretation |
|-------|---------------------|---------|----------------|
| L14 | +0.0526 | 7.08 | +uncertainty → +confidence |
| L15 | +0.0410 | 4.61 | +uncertainty → +confidence |

**Adding** the uncertainty direction **increases** stated confidence. This confirms:
1. The direction points toward "certain" (not "uncertain" despite the name)
2. The direction causally influences confidence expression
3. The negative projection explanation is consistent—ablation effectively adds the direction

## Ruling Out the Inhibition Hypothesis

The "inhibition hypothesis" proposed that uncertainty might suppress confidence (like a brake pedal). This would predict:

| Prediction | What We See | Verdict |
|------------|-------------|---------|
| +uncertainty → -confidence | +uncertainty → +confidence | Contradicted |
| Directions anti-aligned | cos ~0.5 (aligned) | Contradicted |
| Ablation disinhibits | Ablation adds (via negative projection) | Simpler explanation |

**Verdict: Inhibition is ruled out.** The directions are aligned, not opposing. The confidence increase from ablation is explained by negative projection mechanics, not disinhibition.

## Summary

The uncertainty direction (trained on MC logit_gap) is **causally necessary** for calibrated confidence in the meta-task:

1. **It encodes per-sample uncertainty information** - not just a mean signal, but which specific samples the model is uncertain about

2. **Ablating it destroys this differentiation** - all samples get pushed toward the "certain" side (because they had negative projections), breaking the model's ability to express calibrated confidence

3. **The effect is not inhibition** - both directions point toward "confident," and ablation increases confidence via straightforward vector arithmetic

4. **L17 marks a transition** - the simple explanation holds for L14-L16 but breaks down at L17, suggesting a change in how these signals are processed

## Files

- Analysis script: `analyze_uncertainty_confidence_relationship.py`
- Cross-direction causality: `run_cross_direction_causality.py`
- Steering results: `*_steering_confidence_logit_gap_final_mean_diff.png`
- Ablation results: `*_ablation_confidence_uncertainty_logit_gap_mean_diff_final.png`
- Per-sample data: `*_cross_direction_logit_gap_per_sample.npz`
