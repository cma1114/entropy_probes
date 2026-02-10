# Orthogonal Directions Analysis

## Overview

This document describes the methodology for computing orthogonalized confidence direction vectors and analyzing their consistency across datasets.

## Conceptual Framework

### The Problem

When a model predicts its own confidence (self-confidence), two signals may be confounded:

1. **Self-knowledge**: The model's access to its internal uncertainty state
2. **Surface difficulty cues**: Observable features that correlate with question difficulty (e.g., obscure vocabulary, complex syntax)

Similarly, when predicting another model's confidence (other-confidence), the model primarily uses surface difficulty cues (it can't access another model's internal state).

### The Solution: Orthogonalization

We extract two direction vectors:
- **d_self_confidence**: Direction that predicts self-confidence
- **d_other_confidence**: Direction that predicts other-confidence

Then apply Gram-Schmidt orthogonalization:
```
d_self_confidence_unique = d_self_confidence - proj(d_self_confidence, d_other_confidence)
                         = d_self_confidence - cos(d_self_confidence, d_other_confidence) * d_other_confidence

d_other_confidence_unique = d_other_confidence - proj(d_other_confidence, d_self_confidence)
                          = d_other_confidence - cos(d_other_confidence, d_self_confidence) * d_self_confidence
```

This yields:
- **d_self_confidence_unique**: The component of self-confidence prediction that is *orthogonal* to other-confidence prediction
- **d_other_confidence_unique**: The component of other-confidence prediction that is *orthogonal* to self-confidence prediction

### Alternative Approach: Paired Contrast Direction

A complementary approach works directly on paired activations rather than direction vectors:

```
d_selfVother_conf = normalize(mean(self_activation[i] - other_activation[i]))
```

For each question, we have activations from two contexts:
- **Self-confidence trial**: "How confident are you in your answer?"
- **Other-confidence trial**: "How confident would [other model] be?"

Since these are the **same questions** with the **same answer options**, the only difference is the task framing. The contrast direction captures what changes in activation space when the model is asked about itself vs another model.

**Key difference from d_self_confidence_unique:**
- d_self_confidence_unique: Orthogonalizes *direction vectors* (found from high/low confidence samples)
- d_selfVother_conf: Averages *paired activation differences* (same question, different task)

Both aim to isolate self-specific signal, but from different angles.

## Direction Finding Methods

### Mean-Difference Method

```python
# Sort samples by confidence, take top/bottom 25%
sorted_idx = np.argsort(confidences)
low_idx = sorted_idx[:n_group]   # bottom 25%
high_idx = sorted_idx[-n_group:]  # top 25%

# Direction = centroid difference
direction = mean(activations[high_idx]) - mean(activations[low_idx])
direction = direction / norm(direction)
```

### Probe Method (Alternative)

Train a Ridge regression probe to predict confidence from activations. The probe's weight vector (projected back through PCA and standardization) defines the direction.

## Train/Test Split

**Critical for validity:** Directions are found using only training data (80%), then evaluated on held-out test data (20%). Both scripts use identical split parameters (`SEED=42`, `TRAIN_SPLIT=0.8`) ensuring the same samples are held out.

This prevents circularity: the R² values for predictive power are genuine out-of-sample measurements.

## Cross-Dataset Analysis

### Pairwise Similarity

For each pair of datasets, compute cosine similarity between their direction vectors at each layer:

```
cos(d_self_confidence_unique_DatasetA, d_self_confidence_unique_DatasetB)
```

Moderate pairwise similarities (0.3-0.5) with high variance across layers suggest:
- Some layers find similar directions across datasets
- Other layers find dataset-specific directions
- There may be a shared core signal with additive noise

### Consensus Vector

To extract the "common core" shared across all datasets:

```python
# Stack direction vectors from all datasets
vecs = [d_self_confidence_unique_A, d_self_confidence_unique_B, d_self_confidence_unique_C]

# Average and renormalize
consensus = normalize(mean(vecs))
```

**Why averaging is optimal:**

The consensus vector maximizes average cosine similarity to all input vectors:

```
argmax_v sum_i cos(v, d_i) = argmax_v sum_i (v · d_i)
                           = argmax_v v · sum_i(d_i)
                           = normalize(sum_i(d_i))
                           = normalize(N * mean(d_i))
```

### Alignment to Consensus

Compute how well each dataset's direction aligns with the consensus:

```
alignment = cos(d_self_confidence_unique_DatasetA, d_self_confidence_unique_consensus)
```

**Key insight:** If pairwise similarities are moderate but alignment to consensus is high, this indicates:

```
Dataset direction = shared_signal + dataset_noise

Pairwise: cos(signal + noise_A, signal + noise_B) = moderate
          (different noises reduce similarity)

Consensus ≈ signal + (noise_A + noise_B + noise_C)/3 ≈ signal
          (noises cancel when averaged)

Alignment: cos(signal + noise_A, signal) = high
          (noise is small relative to signal)
```

This pattern confirms a genuine universal direction with dataset-specific perturbations.

### Interpreting Alignment Values

| Alignment | Shared Variance | Interpretation |
|-----------|-----------------|----------------|
| 0.9 | 81% | Nearly universal direction |
| 0.8 | 64% | Strong shared signal |
| 0.7 | 49% | Moderate shared signal |
| 0.5 | 25% | Weak shared signal |
| 0.3 | 9% | Mostly dataset-specific |

## Logit Lens Analysis

Project direction vectors through the unembedding matrix to see what vocabulary tokens they encode:

```python
# Apply final layer norm (RMSNorm for Llama)
direction_normed = rms_norm(direction, norm_weight)

# Project through unembedding
logits = direction_normed @ lm_head.T

# Get top-k tokens by softmax probability
probs = softmax(logits)
top_tokens = argsort(probs)[-k:]
```

This reveals semantic content encoded by each direction.

## Direction Types Summary

| Direction | Source | What it captures |
|-----------|--------|------------------|
| **d_self_confidence** | `*_meta_confidence_metaconfdir_directions.npz` | Predicts self-confidence |
| **d_other_confidence** | `*_meta_other_confidence_metaconfdir_directions.npz` | Predicts other-confidence |
| **d_self_confidence_unique** | `*_orthogonal_directions.npz` | d_self_confidence with d_other_confidence projected out |
| **d_other_confidence_unique** | `*_orthogonal_directions.npz` | d_other_confidence with d_self_confidence projected out |
| **d_selfVother_conf** | `*_selfVother_conf_directions.npz` | mean(self_activation - other_activation) for paired questions |
| **d_mc_{metric}** | `*_mc_{metric}_directions.npz` | Predicts MC answer metric (from identify_mc_correlate.py, e.g. logit_gap, entropy) |

## Pipeline

### Step-by-step for multiple datasets

Each script has constants at the top that must be set before running.

#### Step 1: For EACH dataset, run identify_mc_correlate.py

```python
# In identify_mc_correlate.py, set:
DATASET = "TriviaMC_difficulty_filtered"  # or "PopMC_0_difficulty_filtered" or "SimpleMC"
```
```bash
python identify_mc_correlate.py
```

#### Step 2: For EACH dataset, get d_self_confidence

```python
# In test_meta_transfer.py, set:
DATASET = "TriviaMC_difficulty_filtered"  # change for each dataset
META_TASK = "confidence"
FIND_CONFIDENCE_DIRECTIONS = True
```
```bash
python test_meta_transfer.py
```

#### Step 3: For EACH dataset, get d_other_confidence

```python
# In test_meta_transfer.py, set:
DATASET = "TriviaMC_difficulty_filtered"  # change for each dataset
META_TASK = "other_confidence"
FIND_CONFIDENCE_DIRECTIONS = True
```
```bash
python test_meta_transfer.py
```

#### Step 4: Compute orthogonal directions (all datasets at once)

```python
# In compute_orthogonal_directions.py, set:
MODEL_SHORT = "Llama-3.1-8B-Instruct"
ADAPTER = None  # Or path to adapter if used in identify step
DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]
```
```bash
python compute_orthogonal_directions.py
```

#### Step 5: Compute selfVother_conf directions (all datasets at once)

```python
# In compute_contrast_direction.py, set:
MODEL_SHORT = "Llama-3.1-8B-Instruct"
ADAPTER = None  # Or path to adapter if used in identify step
DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]
```
```bash
python compute_contrast_direction.py
```

#### Step 6: Cross-dataset comparison and consensus

```python
# In compare_directions_cross_dataset.py, set:
MODEL_SHORT = "Llama-3.1-8B-Instruct"
DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]
```
```bash
python compare_directions_cross_dataset.py
```

This produces `{model}_consensus_directions.npz`.

#### Step 7: Logit lens on all directions INCLUDING consensus

```python
# In analyze_directions.py, set:
DATASET_FILTER = None  # IMPORTANT: None to include consensus directions
```
```bash
python analyze_directions.py
```

If you set `DATASET_FILTER = "TriviaMC_difficulty_filtered"`, it will only analyze that dataset and skip consensus.

### Quick Reference: Simplified Workflow (Steps 4-7)

Once Steps 1-3 are done for all datasets, the remaining steps are just:

```bash
python compute_orthogonal_directions.py   # d_self_confidence_unique, d_other_confidence_unique
python compute_contrast_direction.py      # d_selfVother_conf
python compare_directions_cross_dataset.py # consensus
python analyze_directions.py               # logit lens
```

Just ensure the `DATASETS` list is the same in all four scripts.

#### Optional: Full causal analysis (takes hours)

```python
# In analyze_introspection_orthogonalization.py, set:
INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC_difficulty_filtered"
```
```bash
python analyze_introspection_orthogonalization.py
```

## Output Files

| File | Contents |
|------|----------|
| `*_meta_confidence_metaconfdir_directions.npz` | d_self_confidence per layer |
| `*_meta_other_confidence_metaconfdir_directions.npz` | d_other_confidence per layer |
| `*_orthogonal_directions.npz` | d_self_confidence_unique, d_other_confidence_unique per layer |
| `*_selfVother_conf_directions.npz` | d_selfVother_conf per layer |
| `*_orthogonal_analysis_results.json` | Predictive power, steering, ablation results |
| `{model}_cross_dataset_comparison.json` | Pairwise similarities, alignment scores for all direction types |
| `{model}_consensus_directions.npz` | Consensus for all direction types |
| `*_direction_analysis.json` | Logit lens results (top-k tokens per layer) |

## Key Findings Template

When running this analysis, report:

1. **Cosine similarity cos(d_self_confidence, d_other_confidence)**: How much do self and other confidence directions overlap? High overlap (>0.8) means limited unique content.

2. **Predictive power of d_self_confidence_unique for self vs other**: Does the unique component of d_self_confidence predict self-confidence (high R²) but NOT other-confidence (R² ≈ 0)?

3. **Cross-dataset consistency**: Alignment to consensus >0.7 suggests universal directions; <0.5 suggests dataset-specific artifacts.

4. **Logit lens interpretation**: What semantic content do the directions encode? Do d_self_confidence_unique and d_other_confidence_unique point to qualitatively different token types?

5. **d_self_confidence_unique vs d_selfVother_conf comparison**: These two approaches should be compared:
   - High cosine similarity → Both methods capture the same signal
   - Low similarity → They capture different aspects
   - If d_selfVother_conf has higher cross-dataset consistency, paired subtraction may be a cleaner method
