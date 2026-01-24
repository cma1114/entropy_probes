# Cross-Dataset Transfer: Observations and Analysis

This document captures key observations from investigating cross-dataset transfer of uncertainty directions.

## Setup

- **Model**: Llama-3.3-70B-Instruct
- **Datasets**: PopMC, SimpleMC, TriviaMC (500 samples each)
- **Metrics**: entropy, logit_gap
- **Methods**: mean_diff (centroid difference), probe (Ridge regression with PCA)
- **Tasks**: D2D (direct-to-direct), D2M (direct-to-meta)

## Key Finding: Direction Degeneracy

**Observation**: Directions trained on different datasets have low cosine similarity (~0.25-0.40) yet achieve similar prediction performance.

From `diagnose_direction_overlap.py` at layer 50 (logit_gap, probe):

| Pair | Cosine Sim | Cross r | Within r | Parallel r | Orthogonal r |
|------|-----------|---------|----------|------------|--------------|
| PopMC vs SimpleMC | 0.38 | 0.74 | 0.81 | 0.74 | 0.79 |
| PopMC vs TriviaMC | 0.40 | 0.96 | 0.98 | 0.96 | 0.97 |
| SimpleMC vs TriviaMC | 0.25 | 0.93 | 0.98 | 0.93 | 0.97 |

**Key insight**: Both the parallel component (projection of dir_B onto dir_A) AND the orthogonal component (remainder) predict well. This means:

1. The uncertainty signal is **not captured by a single direction** but is spread across a subspace
2. Multiple different directions can capture the signal equally well
3. Ridge regression / mean_diff finds *a* good direction, not *the* direction

**Evidence**: Combining directions gives zero improvement over single directions:
- Within direction alone: r = 0.97
- Optimal combination of all directions: r = 0.97
- Improvement: 0.00

This confirms the signal is fully captured by any single well-trained direction - there's no additional information in other directions.

## D2D Transfer Performance

Cross-dataset transfer (train on A, test on B) achieves ~95% of within-dataset performance for logit_gap:

| Metric | Method | Cross r | Within r | Ratio |
|--------|--------|---------|----------|-------|
| logit_gap | probe | 0.85-0.96 | 0.90-0.98 | ~0.95 |
| logit_gap | mean_diff | 0.80-0.94 | 0.85-0.97 | ~0.94 |
| entropy | probe | 0.60-0.80 | 0.65-0.85 | ~0.92 |
| entropy | mean_diff | variable | variable | variable |

**logit_gap transfers better than entropy** across all comparisons.

## D2M Transfer Performance

Transfer to meta-task (confidence/delegation) shows:
- Cross and within achieve similar performance (~r=0.5 in later layers)
- Neither dramatically outperforms the other
- This is consistent with the degeneracy finding: any reasonable direction works

## Statistical Considerations

### Wide Null Distributions for D2M

D2M permutation nulls have much wider variance than D2D:
- D2D null_std: ~0.10-0.20
- D2M null_std: ~0.40-0.50

This is expected because:
1. Meta activations have different structure than MC activations
2. Confidence is a different target than entropy/logit_gap
3. Signal-to-noise ratio is lower for the meta-task

### Sample Size Effects

With n=100 test samples (20% of 500):
- Standard error of correlation: SE ≈ sqrt((1-r²)/(n-2)) ≈ 0.05-0.10 for moderate r
- Small differences between cross and within (e.g., 0.81 vs 0.78) are not statistically significant

## Float16 Precision Issues

Activations are stored as float16 (from model's default dtype). This caused:
1. **Arithmetic overflow**: Operations like `.mean()` and matrix multiplication can overflow when summing many float16 values
2. **Symptom**: `std = inf`, RuntimeWarnings about divide by zero
3. **Fix**: Cast to float32 at the start of any function doing arithmetic on activations

## Interpretation

### Why do different directions work equally well?

The uncertainty signal appears to be **low-rank but not 1-dimensional**. In high-dimensional activation space (~8192 dims), many directions correlate with uncertainty. Ridge regression finds a direction that predicts well, but it's not unique.

Geometrically: imagine uncertainty as a hyperplane in activation space. Any vector with a component along this hyperplane will correlate with uncertainty. Different training sets find different vectors, all with components along the hyperplane.

### Implications for interpretability

1. **Caution with direction cosine similarity**: Low similarity doesn't mean directions capture different signals
2. **Probing finds *a* direction, not *the* direction**: Don't over-interpret the specific direction found
3. **Transfer works because the signal is robust**: The uncertainty representation generalizes across datasets because multiple access routes exist

### Implications for steering/ablation

If the signal lives in a subspace rather than a single direction:
1. **Ablating one direction may not fully remove the signal** - other directions in the subspace remain
2. **Steering along one direction may have limited effect** - the model can use other directions
3. **May need to ablate/steer along multiple directions** or identify the full subspace

## Open Questions

1. **What is the dimensionality of the uncertainty subspace?** Could use PCA on directions from many random seeds/datasets to estimate.

2. **Why does logit_gap transfer better than entropy?** Possibly because logit_gap is more linearly related to activations, while entropy has nonlinearities.

3. **Is the degeneracy specific to uncertainty, or general to probing?** Would other targets (sentiment, factuality) show similar patterns?

4. **How does this affect causal claims?** If ablation only removes one of many redundant directions, negative ablation results don't rule out causal involvement.
