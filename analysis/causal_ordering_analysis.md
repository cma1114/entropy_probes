# Causal Ordering Analysis: Does MC Uncertainty Drive Meta-Judgments?

## Hypothesis

If the model's MC uncertainty causally influences its meta-judgments (confidence ratings, delegation decisions), then:

1. MC uncertainty signals should appear in **earlier** layers (input processing)
2. Meta output entropy should appear in **middle** layers (decision computation)
3. Stated confidence should appear in **later** layers (output generation)

We test this by comparing layer-by-layer R² curves for four signal types extracted from meta-task activations.

## Signals Measured

| Signal | Description | What it tests |
|--------|-------------|---------------|
| **D→M transfer** | MC uncertainty direction (from MC task) projected onto meta activations | Does the *same* direction transfer? |
| **mcuncert** | Direction trained on meta activations to predict MC uncertainty | Is MC uncertainty encoded *somewhere* in meta activations? |
| **metaentropy** | Direction predicting entropy over meta-task output options | When does the model "decide" its meta output? |
| **confdir** | Direction predicting stated confidence | When does confidence get computed? |

## Results (TriviaMC_difficulty_filtered, confidence task)

### Peak Layers

| Signal | Probe Peak | Mean-diff Peak | Best R² |
|--------|------------|----------------|---------|
| D→M transfer | 14 | 16 | 0.36 |
| mcuncert | 19 | 15 | 0.48 |
| metaentropy | 30 | 29 | 0.90 |
| confdir | 30 | 17 | 0.99 |

MC signals average peak: **layer 16**
Meta signals average peak: **layer 26.5**

At first glance, this appears to support the causal hypothesis (MC peaks before meta). But the **shape of the curves** tells a different story.

### Curve Shapes

**D→M transfer (probe):**
- Rises gradually from layer 0
- Peaks at layer 14-17 (R² ~ 0.35)
- **Falls sharply** to R² ~ 0.08 by layer 30

**confdir (probe):**
- Near zero until layer 8
- Rises steeply from layer 12-17 (R² goes 0.55 → 0.97)
- **Plateaus** at R² ~ 0.98-0.99 through layer 31

**Key observation:** D→M transfer and confdir both "turn on" around **the same layers (12-17)**, not sequentially. Then D→M **disappears** in exactly the layers where confdir is strongest.

### Layer-by-Layer Detail

| Layer | D→M (probe) | confdir (probe) | Interpretation |
|-------|-------------|-----------------|----------------|
| 12 | 0.12 | 0.55 | Both rising |
| 14 | **0.36** | 0.82 | D→M at peak |
| 17 | 0.33 | **0.97** | confdir nearly saturated |
| 20 | 0.10 | 0.98 | D→M falling, confdir stable |
| 25 | 0.07 | 0.98 | D→M nearly gone |
| 30 | 0.03 | 0.99 | D→M gone, confdir perfect |

## Interpretation

### What we'd expect if MC uncertainty drives confidence:

1. D→M transfer should appear **before** confdir
2. D→M should **stay elevated** in layers where confdir is being computed (if the model is "reading" the MC uncertainty direction)

### What we actually see:

1. D→M and confdir turn on **simultaneously** (both around layer 12-14)
2. D→M **falls off** in late layers while confdir stays at R² ~ 0.99
3. The model "forgets" the MC uncertainty direction in exactly the layers where confidence is strongest

### The mcuncert puzzle:

mcuncert (direction trained on meta activations to predict MC uncertainty) stays elevated at R² ~ 0.45 through late layers. This shows the model *does* have MC uncertainty information available in its meta-task activations.

But this is a **different direction** than D→M transfer. The model encodes MC uncertainty in meta activations, but not using the same geometric representation it uses in the MC task. This suggests parallel encoding rather than causal reuse.

## Conclusions

1. **No evidence of causal flow from MC → confidence.** The MC uncertainty direction (D→M) disappears in the layers where confidence is computed.

2. **Simultaneous onset, not sequential.** Both signals appear around layers 12-17, inconsistent with MC uncertainty being an upstream cause.

3. **Parallel representations.** The model encodes MC uncertainty in meta activations (mcuncert R² ~ 0.45) but through a different direction than the original MC task. This suggests the model reconstructs or re-encodes uncertainty information rather than directly reading it.

4. **Confidence is highly linear.** confdir reaches R² ~ 0.99, meaning stated confidence is almost entirely determined by a linear projection of late-layer activations. But this linear structure doesn't connect to the MC uncertainty direction.

### Bottom line:

The model's confidence output appears to be generated through a **separate pathway** that correlates with MC uncertainty (because both reflect underlying question difficulty) but does **not** causally read the MC uncertainty representation. This argues against genuine introspection and in favor of parallel feature-based computation.

## Future Directions

1. **Causal intervention:** Ablate the D→M direction in middle layers and measure effect on confidence. If confidence is unaffected, this confirms the parallel pathway hypothesis.

2. **Cross-task comparison:** Run the same analysis on delegate task (where meta output is binary Answer/Delegate) to see if the pattern holds.

3. **Steering:** Can we steer confidence by adding/subtracting the MC uncertainty direction? If not, further evidence against causal connection.

---

*Analysis based on: TriviaMC_difficulty_filtered, meta_task=confidence, model=Llama-3.1-8B-Instruct*
*Data: outputs/TriviaMC_difficulty_filtered_meta_confidence_causal_ordering.json*
