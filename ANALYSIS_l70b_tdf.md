# Analysis of Llama-70B Introspection Experiments on TriviaMC Difficulty-Filtered

**Model**: Llama-3.3-70B-Instruct
**Dataset**: TriviaMC_difficulty_filtered (250 correct + 250 incorrect questions)
**Results directory**: `outputs/l70b_tdf/`

---

## Overview

This analysis examines the results from running the full introspection pipeline. The experiments test whether the model genuinely introspects on its uncertainty during meta-cognitive tasks (confidence reporting, delegation decisions) or uses different features.

---

## Experiments Run

### 1. Direct Task: MC Question Answering (`identify_mc_correlate.py`)
- **Purpose**: Extract activations while model answers MC questions, find directions that predict output uncertainty
- **Metrics probed**: entropy, logit_gap
- **Methods**: ridge regression probe + mean_diff (centroid difference)
- **Output**: Uncertainty directions per layer

### 2. Answer Direction Finding (`identify_mc_answer_correlate.py`)
- **Purpose**: Find directions that predict which answer (A/B/C/D) the model chose
- **Method**: 4-class classification probe + centroid method
- **Output**: Answer directions per layer

### 3. Meta-Transfer Testing (`test_meta_transfer.py`)
- **Purpose**: Test if uncertainty directions found on direct task transfer to meta-tasks
- **Tasks tested**:
  - `confidence`: Model reports confidence as 0-10 rating
  - `delegate`: Model chooses to answer or delegate to human
  - `other_confidence`: Model estimates human difficulty for the question
- **Output**: Transfer R² per layer

### 4. Confidence Direction Finding (`identify_confidence_correlate.py`)
- **Purpose**: Find directions that predict stated confidence from meta-task activations
- **Tasks**: delegate, confidence
- **Output**: Confidence directions per layer

### 5. Direction Comparison (`compare_direction_types.py`)
- **Purpose**: Compare cosine similarity between direction types (uncertainty vs confidence)
- **Output**: Per-layer cosine similarities

### 6. Cross-Prediction (`cross_predict_confidence.py`)
- **Purpose**: Test if self-confidence probes predict other-confidence and vice versa
- **Output**: Cross-domain R² values

---

## Key Results

### A. Uncertainty Probing (Direct Task)

**Entropy prediction R² by layer (probe method)**:
- Early layers (0-10): R² ~ -0.2 to -0.08 (poor, negative = worse than mean prediction)
- Middle layers (30-40): R² improves but still slightly negative
- Best performance around layers 35-50

The model's internal representations do encode uncertainty, but the encoding becomes clearer in later layers.

### B. Answer Classification (What Letter Did the Model Choose?)

**Test accuracy by layer (4-class classification)**:

| Layers | Probe Accuracy | Centroid Accuracy |
|--------|----------------|-------------------|
| 0-20   | 26-43%         | 29-37%           |
| 35     | **89%**        | 69%               |
| 40-55  | 90-92%         | 70-78%           |
| 76-79  | **96-97%**     | 77-81%           |

**Interpretation**: By layer 35, the model has largely decided which answer to give (89% accuracy). By the final layers, this is near-deterministic (97%). The centroid method never reaches the probe's performance, suggesting the answer representation is more complex than simple cluster centroids.

### C. Meta-Transfer: Do Uncertainty Directions Predict Meta-Task Confidence?

| Meta-Task | Best Layer | Best R² | Best Pearson |
|-----------|------------|---------|--------------|
| confidence | 30 | **-0.09** | 0.32 |
| delegate | 79 | **-0.16** | 0.11 |
| other_confidence | 35 | **-0.09** | 0.32 |

**Interpretation**: Transfer is **WEAK**. The uncertainty directions found on the direct task do NOT strongly predict confidence on meta-tasks. Negative R² means the transferred direction predicts worse than just predicting the mean. However, the positive Pearson correlations (~0.32 for confidence tasks) suggest there IS some signal, just not well-calibrated.

### D. Confidence Direction Finding (Meta-Task Specific)

#### Delegate task - Predicting P(answer) vs P(delegate)

This trains a probe **directly on delegate-task activations** to predict the model's answer/delegate probability.

| Layer | Test R² | Test Pearson | 95% CI |
|-------|---------|--------------|--------|
| 0 | 0.43 | 0.66 | [0.24, 0.57] |
| 14 | **0.69** | **0.84** | [0.61, 0.75] |
| 23 | **0.80** | **0.90** | [0.75, 0.85] |
| 28 | 0.82 | 0.91 | [0.77, 0.87] |
| 31 | **0.86** | **0.94** | [0.79, 0.89] |

The delegation decision is **extremely well-encoded** - a linear probe explains 86% of variance at layer 31.

#### Confidence task - Predicting stated confidence (0-10 scale)

- Test R² mostly negative (-0.15 to +0.02)
- Poor prediction across all layers
- The stated confidence rating is NOT well-encoded linearly

**Key insight**: The delegate task (behavioral binary choice) is MUCH better predicted from activations than the explicit confidence rating (arbitrary 0-10 scale). This may be because delegation is a more "natural" behavioral variable.

### E. Direction Comparison: How Similar Are Different Direction Types?

| Comparison | Mean Cosine | Max |Cosine| Best Layer |
|------------|-------------|----------------|------|
| Uncertainty vs Delegate-Confidence | **0.05** | 0.16 | 35 |
| Uncertainty vs Confidence-Confidence | **0.40** | **0.62** | 38 |
| Uncertainty vs Other-Confidence | 0.04 | 0.50 | 0 |
| Delegate vs Confidence | 0.15 | 0.32 | 34 |
| Delegate vs Other | 0.08 | 0.30 | 33 |
| Self-Confidence vs Other-Confidence | **0.44** | **0.82** | 28 |

**Key findings**:

The **uncertainty direction** is:
- **Moderately aligned** with confidence-confidence (cos=0.40 mean, 0.62 max)
- **Nearly orthogonal** to delegate-confidence (cos=0.05 mean)
- **Nearly orthogonal** to other-confidence (cos=0.04 mean)

The **self-confidence and other-confidence directions** are **highly aligned** (cos=0.44 mean, 0.82 max at layer 28).

### F. Cross-Prediction: Can Self-Confidence Probe Predict Other-Confidence?

From `confidence_cross_prediction.json`:

| Prediction Type | R² Range |
|-----------------|----------|
| self_to_self | -0.04 to +0.15 (modest) |
| self_to_other | **-5.4 to -5.5** (catastrophic failure) |
| other_to_self | **-18 to -19** (catastrophic failure) |
| other_to_other | -0.15 to +0.15 (modest) |

**Interpretation**: Despite the directions being similar (cos~0.8), the probes trained on one task do NOT transfer to the other. This could be due to:
1. Different scales/distributions of the confidence values
2. Different input feature spaces despite similar directions
3. The probes learning task-specific calibration

---

## Implications for Introspection

### What the Model IS Doing

1. **The model encodes which answer it will give** extremely well (97% accuracy at late layers). By layer 35, this is already 89% determined.

2. **The model encodes something related to uncertainty**, but the encoding is noisy early on and becomes clearer in middle-to-late layers.

3. **During the delegate task**, the model's activations strongly predict whether it will answer or delegate (R²=0.86, r=0.94 at best layers). This decision is well-represented internally.

4. **Self-confidence and other-confidence use similar directions** (cos~0.82), suggesting the model uses a shared "difficulty/confidence" concept for both itself and others.

### Evidence AGAINST Simple Introspection

1. **Weak transfer from uncertainty to meta-confidence**: If the model simply read out its uncertainty when reporting confidence, we'd expect strong transfer. Instead, R² is negative.

2. **Uncertainty and confidence directions are only moderately aligned** (cos=0.4-0.6 in middle layers). There's shared information, but also substantial independence.

3. **Uncertainty and delegate directions are nearly orthogonal** (cos=0.05). The delegation decision appears to use DIFFERENT features than raw output uncertainty.

4. **Cross-prediction failure**: Even though self/other confidence directions are similar, probes don't transfer, suggesting context-specific processing.

### Possible Interpretations

**Hypothesis 1: Partial Introspection**
The model accesses its uncertainty representation as ONE input among many when forming confidence judgments. The moderate correlation (0.4-0.6) reflects genuine but incomplete information flow from uncertainty to confidence.

**Hypothesis 2: Separate Confidence Heuristics**
The model may use surface features (question difficulty markers, domain familiarity cues, answer obviousness) to form confidence, which happen to correlate with actual uncertainty but aren't derived from it.

**Hypothesis 3: Different Abstraction Levels**
The uncertainty direction captures output-level statistics (entropy over tokens), while the confidence direction captures a more abstract "difficulty" concept that includes but isn't limited to token-level uncertainty.

### The Delegate Task is Different

The near-zero correlation between uncertainty and delegate-confidence directions suggests the delegation decision uses fundamentally different features. Possibilities:
- Social/pragmatic reasoning about when to defer
- Risk assessment beyond pure uncertainty
- Learned heuristics about when humans prefer answers vs. deferrals

---

## Summary Table

| Finding | Implication |
|---------|-------------|
| Weak uncertainty→confidence transfer (R² < 0) | Model doesn't simply "read out" uncertainty |
| Moderate direction alignment (cos=0.4-0.6) | Some shared information, but not identity |
| High self/other confidence alignment (cos=0.82) | Shared "difficulty" concept across perspectives |
| Delegate uses orthogonal direction (cos=0.05) | Different features for delegation decision |
| Excellent answer prediction (97%) | Model's choice is crystallized by late layers |
| Delegate R²=0.86, Confidence R²<0 | Behavioral decisions better encoded than scale ratings |

---

## Conclusion

The evidence suggests the model's confidence reporting involves more than direct introspection on output uncertainty. There's shared information (moderate cosine similarity), but the confidence direction captures additional or different features.

The delegation task is particularly interesting: it's extremely well-predicted from activations (R²=0.86) but uses a direction that's **orthogonal** to the uncertainty direction. This suggests the model has learned a separate computational pathway for deciding when to defer - one that isn't based on reading out token-level uncertainty.

The failure of cross-prediction between self-confidence and other-confidence (despite highly similar directions) suggests that even when the model uses similar representational geometry, the specific mapping from activations to outputs is task-specific.
