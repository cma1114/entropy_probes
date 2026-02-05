# Analysis: Why Ablating Uncertainty Increases Confidence Projection

**Date**: 2024-02-03
**Model**: Llama-3.1-8B-Instruct
**Dataset**: TriviaMC (difficulty filtered)

## The Puzzle

When ablating the **uncertainty direction** (trained on logit_gap) at layer L14, the projection of downstream activations onto the **confidence direction** (trained on stated confidence) **increases** rather than decreases.

This was initially puzzling because:
- If uncertainty "inhibits" confidence, ablating it should increase confidence (observed)
- But the directions should then be anti-aligned (NOT observed)

## Key Results

### Behavioral Effects (Consistent with Causal Role)

| Test | Effect | Interpretation |
|------|--------|----------------|
| Steering +uncertainty | +confidence (slope=0.053, L14, z=7.08) | Sufficiency: adding uncertainty signal increases stated confidence |
| Ablation uncertainty | -calibration (Δr=-0.105, L15) | Necessity: removing direction degrades calibration |

Both behavioral tests confirm the uncertainty direction is causally important.

### Direction Relationship (Against Inhibition)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sample projection correlation | r = +0.90 to +0.96 (L13-17) | Strongly POSITIVE (not negative) |
| Direction cosine similarity | cos = +0.41 to +0.57 (L13-17) | ALIGNED (not anti-aligned) |

The inhibition hypothesis predicted negative correlation and anti-alignment. **It is ruled out.**

### Cross-Direction Ablation (The Artifact)

| Ablation | Raw Δ | Normalized Δ | Reduction | Significant? |
|----------|-------|--------------|-----------|--------------|
| L14→L15 | +0.015 | +0.002 | 89% | Raw: Yes, **Norm: No** |
| L14→L17 | +0.053 | +0.005 | 91% | Raw: Yes, **Norm: No** |
| L14→L20 | +0.074 | +0.006 | 92% | Raw: Yes, **Norm: No** |
| L14→L25 | +0.102 | +0.005 | 96% | Raw: Yes, **Norm: No** |
| L14→L31 | +0.239 | +0.003 | **99%** | Raw: Yes, **Norm: No** |

After normalizing by activation norm:
- All effects become **non-significant**
- Effect reduction: **89-99%**
- Reduction grows with layer distance (as predicted by LayerNorm amplification)

## The Explanation: LayerNorm Amplification

### Mechanism

When ablating at layer N and measuring projection at layer M > N:

1. **Ablation removes a component**: `x' = x - (x·u)u`
   - The activation now has slightly different magnitude and direction

2. **LayerNorm rescales**: Each subsequent transformer layer normalizes to fixed variance
   - This amplifies or dampens components to maintain standard scale

3. **Projection changes**: The confidence projection is computed as `x'·c`
   - Because of rescaling dynamics, this can increase even when the ablated component was positively aligned

4. **Effect accumulates**: More layers between ablation and measurement = more LayerNorm operations = larger artifact

### Evidence

- Norm ratio (ablated/baseline) ≈ 0.995-1.002 (barely changes)
- But after proper normalization, the "effect" disappears
- Effect reduction grows from 89% (gap=1) to 99% (gap=17)

## Conclusions

### 1. The puzzle is resolved

The cross-direction ablation effect is **not a real causal relationship**. It's an artifact of how projections scale with activation norm through multiple LayerNorm operations.

### 2. The uncertainty direction IS causally important

The behavioral effects are real:
- Steering +uncertainty → +confidence (sufficiency)
- Ablating uncertainty → degraded calibration (necessity)

The internal projection paradox was a measurement artifact, not evidence against causality.

### 3. Why directions are positively aligned

Both directions point toward "more confident":
- **Uncertainty (logit_gap)**: Higher logit_gap = larger margin = more certain about answer
- **Confidence direction**: Learned from stated confidence, which correlates with logit_gap

They capture **overlapping variance** (cos=0.4-0.6), not opposite constructs. This makes sense: when the model is uncertain (low logit_gap), it should and does state lower confidence.

### 4. Methodological insight

**Cross-direction ablation projections are confounded by activation norm changes.**

Best practices:
- Always report normalized projections alongside raw projections
- Check if effects survive normalization
- Prefer steering (additive intervention) over ablation (projective intervention) when possible
- Be skeptical of cross-layer projection changes after ablation

## Files

- Analysis script: `analyze_uncertainty_confidence_relationship.py`
- Analysis results: `*_uncertainty_confidence_analysis.json`
- Cross-direction results: `*_cross_direction_logit_gap_results.json`
- Steering results: `*_steering_confidence_logit_gap_*.png`
- Ablation results: `*_ablation_confidence_uncertainty_logit_gap_*.png`
