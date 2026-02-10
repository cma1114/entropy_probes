# Self vs Other Metamcuncert Analysis

## Summary

Comparing metamcuncert directions between self-confidence and other-confidence meta-tasks reveals **method-dependent divergence**. However, **causal validation shows only mean_diff has behavioral impact** — the probe's "unique" component is predictive but not causal, suggesting surface cues dominate.

## Experimental Setup

- **Model**: Llama-3.1-8B-Instruct
- **Dataset**: TriviaMC_difficulty_filtered (500 questions, balanced correct/incorrect)
- **Comparison**: cosine(d_meta_mc_uncert_self, d_meta_mc_uncert_other) per layer

Where:
- `d_meta_mc_uncert_self`: Direction in self-confidence activations predicting MC uncertainty
- `d_meta_mc_uncert_other`: Direction in other-confidence activations predicting MC uncertainty

## Key Finding: Method-Dependent Divergence

The probe and mean_diff methods tell **different stories**:

| Comparison | Early Layers (0-10) | Late Layers (20+) | Mean |
|------------|---------------------|-------------------|------|
| probe ↔ probe | ~0.75 | ~0.38 | 0.54 |
| probe ↔ mean_diff | ~0.43 | ~0.15 | 0.28 |
| mean_diff ↔ probe | ~0.45 | ~0.10 | 0.27 |
| mean_diff ↔ mean_diff | ~0.81 | ~0.91 | 0.86 |

### What Each Method Captures

**Mean_diff method**: Captures the direction between extreme groups (top 25% vs bottom 25% uncertainty). High alignment (~0.9) across all layers means both self and other tasks respond similarly to the *contrast between easy and hard questions*. This IS the surface cue signal.

**Probe method**: Optimized to predict MC uncertainty from the full distribution of activations. Low alignment in late layers (~0.1-0.15) means the full predictive signal diverges between self and other, even though both achieve similar R².

## Interpretation

This pattern suggests **surface cues + genuine introspection**:

1. **Early layers (shared)**: Both tasks detect question difficulty - the obvious surface features. High cosine across methods.

2. **Late layers (divergent for probes)**: The mean_diff signal (extremes) stays aligned, but the probe signal (full prediction) diverges substantially.

This is consistent with:
> "The model uses surface cues (shared, captured by mean_diff), but self-confidence ALSO activates a unique component (captured by probe) that other-confidence doesn't."

## Layer 13-14 Transition

Notably, layer 13-14 shows a transition point:
- Mean_diff alignment jumps from ~0.7 to ~0.9 (surface cue signal strengthens)
- Probe alignment drops from ~0.5 to ~0.3 (unique self-confidence component emerges)

This may indicate where the model transitions from feature extraction to decision-making, with self-confidence accessing an additional internal state at this boundary.

## Causal Validation (2026-02-09)

### Key Finding: Only Mean_diff is Causal

Ablation experiments reveal a critical dissociation:

| Direction | Predictive (R²) | Causal (ablation effect) |
|-----------|-----------------|--------------------------|
| mean_diff | Yes | **Yes** - large behavioral impact |
| probe | Yes | **No** - no behavioral impact |

This is the decisive test. The probe direction diverges between self/other and achieves good R², but ablating it doesn't affect behavior. The mean_diff direction is shared with other-confidence (surface cues) AND is what actually drives behavior.

### Implications

1. **Predictive ≠ Causal**: High R² doesn't prove a direction is used by the model
2. **The "unique" self-confidence component is noise**: It correlates with MC uncertainty but isn't read out
3. **Surface cues dominate**: The shared mean_diff signal (question difficulty) is the causal pathway

### Revised Interpretation

The original "mixed model" interpretation was premature. The causal data suggests:

> The model's confidence reports are driven by surface cue detection (question difficulty), not by genuine introspection on internal uncertainty. The probe method finds spurious correlations that don't drive behavior.

This doesn't rule out introspection entirely, but the evidence here points to surface cues as the primary mechanism.

## Implications (Correlational)

| Hypothesis | Prediction | Observed (correlational) | Observed (causal) |
|------------|------------|--------------------------|-------------------|
| Pure surface cues | High cosine for both methods | ❌ Probe diverges | ✅ Only mean_diff causal |
| Pure introspection | Low cosine for both methods | ❌ Mean_diff stays high | ❌ |
| Mixed model | Method-dependent divergence | ✅ Observed | ❌ Probe not causal |

**Conclusion**: Correlational analysis suggested a mixed model, but causal validation supports **surface cues as the dominant mechanism**.

## Raw Data (2026-02-09)

### Probe ↔ Probe
```
Layer     Cosine
L0         0.858
L1         0.851
L2         0.856
L3         0.841
L4         0.792
L5         0.759
L6         0.739
L7         0.694
L8         0.721
L9         0.611
L10        0.552
L11        0.575
L12        0.592
L13        0.595
L14        0.547
L15        0.434
L16        0.429
L17        0.411
L18        0.354
L19        0.379
L20        0.373
L21        0.396
L22        0.421
L23        0.381
L24        0.357
L25        0.350
L26        0.366
L27        0.393
L28        0.380
L29        0.365
L30        0.394
L31        0.423
```

### Mean_diff ↔ Mean_diff
```
Layer     Cosine
L0         0.866
L1         0.889
L2         0.804
L3         0.836
L4         0.831
L5         0.829
L6         0.792
L7         0.795
L8         0.786
L9         0.755
L10        0.727
L11        0.681
L12        0.714
L13        0.903
L14        0.921
L15        0.877
L16        0.896
L17        0.898
L18        0.905
L19        0.906
L20        0.903
L21        0.913
L22        0.909
L23        0.905
L24        0.903
L25        0.910
L26        0.915
L27        0.920
L28        0.919
L29        0.918
L30        0.916
L31        0.895
```

## Next Steps

1. ~~**Causal validation**~~: ✅ DONE - Only mean_diff is causal, probe is not

2. **Orthogonalization** (less urgent given causal results): Decompose directions to confirm the unique component has no causal role

3. **Cross-model comparison**: Does this pattern (predictive but non-causal probe divergence) replicate in other models?

4. **Alternative introspection tests**: Look for other signatures of introspection that survive causal validation
