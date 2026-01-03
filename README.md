# Introspection Experiments

This repository contains tools for studying whether language models can introspect on their own uncertainty. The core question: when a model chooses to answer or delegate/pass, or reports confidence, is it actually accessing internal representations of its own uncertainty?

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full introspection experiment (MC questions + meta-judgments)
python run_introspection_experiment.py --metric logit_gap

# Run steering experiments
python run_introspection_steering.py --metric logit_gap
```

## Multi-Metric Uncertainty System

The framework supports multiple uncertainty metrics, all computed in a single forward pass:

### Probability-based metrics (nonlinear)
- **entropy**: Shannon entropy `-sum(p * log(p))` — higher = more uncertain
- **top_prob**: `P(argmax)` — probability of most likely answer
- **margin**: `P(top) - P(second)` — gap between top two probabilities

### Logit-based metrics (linear, recommended for probes)
- **logit_gap**: `z(top) - z(second)` — logit gap between top two (invariant to temperature)
- **top_logit**: `z(top) - mean(z)` — centered top logit

**Why logit-based metrics?** Linear probes train on activations, which are linearly transformed into logits. Logit-based targets are better aligned with what linear probes can learn, often yielding higher R² scores.

### CLI Usage

All main scripts accept `--metric` to select which metric to probe:

```bash
python mc_entropy_probe.py --metric logit_gap        # Probe logit_gap (default)
python mc_entropy_probe.py --metric entropy          # Probe entropy

python run_introspection_experiment.py --metric logit_gap
python run_introspection_probe.py --metric entropy
python run_introspection_steering.py --metric logit_gap
```

**Note:** All metrics are computed and saved regardless of which one you probe. You can re-run with `--plot-only` (where supported) to train probes on different metrics without re-extracting activations.

---

## The Analyses

### 1. Is uncertainty predictable from activations? (General case)

**Scripts:** `build_nexttoken_dataset.py` → `nexttoken_entropy_probe.py`

Tests whether next-token entropy (over the full vocabulary) is linearly decodable from activations on diverse text (Wikipedia, code, web crawl).

```bash
# Step 1: Build stratified dataset with diverse text
python build_nexttoken_dataset.py

# Step 2: Train entropy probes
python nexttoken_entropy_probe.py
```

**Outputs:** (prefixed with model name, e.g., `Llama-3.1-8B-Instruct_nexttoken_`)
- `*_samples_raw.json` - Raw samples with entropies (before stratification)
- `*_entropy_dataset.json` - Stratified dataset (even across entropy deciles)
- `*_activations.npz` - Extracted activations by layer
- `*_entropy_probe.json` - Probe R² by layer
- `*_entropy_probe.png` - Visualization

This is the most general test: can we decode entropy from activations on arbitrary text?

### 2. Is uncertainty predictable from activations? (Factual MC case)

**Script:** `mc_entropy_probe.py`

Tests whether uncertainty metrics over MC answer options (A/B/C/D) are predictable from activations on factual questions.

```bash
python mc_entropy_probe.py --metric logit_gap
python mc_entropy_probe.py --metric entropy
```

**Outputs:** (prefixed with model, dataset, and metric, e.g., `Llama-3.1-8B-Instruct_SimpleMC_mc_logit_gap_`)
- `*_activations.npz` - Extracted activations + all metrics (reusable)
- `*_dataset.json` - Questions with uncertainty values
- `*_results.json` - Probe R² by layer
- `*_directions.npz` - Probe directions for steering
- `*_results.png` - Visualization

### 3. Does the uncertainty probe transfer to meta-judgments? (Direct→Meta transfer)

**Script:** `run_introspection_experiment.py`

The core introspection test. For each question:
1. Runs direct MC prompt, extracts activations and uncertainty metrics
2. Runs meta prompt ("How confident are you?"), extracts activations and stated confidence
3. Trains probe: direct activations → selected metric
4. Tests transfer: does direct-trained probe work on meta activations?

```bash
python run_introspection_experiment.py --metric logit_gap
python run_introspection_experiment.py --metric entropy
```

**Meta-judgment task types:** Configure `META_TASK` at the top of the script:

- `META_TASK = "confidence"` (default) - Explicit confidence rating on S-Z scale (<5% to >90%)
- `META_TASK = "delegate"` - Answer vs Delegate choice; confidence = P(Answer)

The delegate task asks "Would you answer this yourself or delegate to a teammate who answers 50% correctly?" This operationalizes confidence as the model's willingness to answer rather than an explicit self-report.

**Outputs:** (prefixed with model, dataset, and metric)
- `*_paired_data.json` - Questions, all metrics, confidences (shared across metrics)
- `*_direct_activations.npz` - Direct prompt activations + all metrics
- `*_meta_activations.npz` - Meta prompt activations
- `*_{metric}_results.json` - Probe performance by layer for selected metric
- `*_{metric}_directions.npz` - Probe directions for steering
- `*_results.png` - Visualization

**Key metrics:**
- `direct_to_direct R²` - Can we predict the metric from direct activations? (sanity check)
- `direct_to_meta R²` - Does the probe transfer to meta activations? (introspection test)
- `behavioral correlation` - Does stated confidence correlate with actual uncertainty?

### 4. Introspection score probe (Meta→Calibration)

**Script:** `run_introspection_probe.py`

Trains a probe on meta activations to predict the introspection score:

```
introspection_score = -metric_z × confidence_z
```

- Positive when aligned (high uncertainty + low confidence, or low uncertainty + high confidence)
- Negative when misaligned

```bash
python run_introspection_probe.py --metric logit_gap
python run_introspection_probe.py --metric entropy
```

**Sign convention:** For non-entropy metrics (where HIGH value = HIGH confidence), the sign is automatically flipped so that negative correlation always indicates introspective behavior.

**Outputs:**
- `*_{metric}_probe_results.json` - Probe metrics with permutation tests
- `*_{metric}_probe_directions.npz` - Introspection directions for steering

### 5. Steering/ablation with probe directions

**Script:** `run_introspection_steering.py`

Tests whether steering along a probe direction affects the model's confidence-uncertainty alignment:
- Loads probe directions (uncertainty or introspection score)
- Adds scaled direction to activations during meta prompts
- Measures change in alignment (does steering make confidence match uncertainty better?)
- Compares to random orthogonal control directions

```bash
python run_introspection_steering.py --metric logit_gap
```

**Configuration options:**

1. `DIRECTION_TYPE` - Which probe direction to use:
   - `"entropy"` - Uses metric probe directions from `run_introspection_experiment.py`
   - `"introspection"` - Uses introspection score directions from `run_introspection_probe.py`
   - `"shared"` - Uses shared MC direction from `analyze_shared_unique.py`

2. `META_TASK` - Which meta-judgment task to run steering on:
   - `"confidence"` (default) - Explicit confidence rating
   - `"delegate"` - Answer vs Delegate choice

**Layer selection logic:** When `STEERING_LAYERS = None` (auto-select):

| Direction Type | Selection Criterion |
|----------------|---------------------|
| `"shared"` | Layers where direct→meta R² ≥ `META_R2_THRESHOLD` (default 0.25) |
| `"entropy"` | Layers where direct→meta R² > 0.1 AND direct→direct R² > 0.05 |
| `"introspection"` | Layers with significant p < 0.05, plus best layer |

**Outputs:**
- `*_steering_{direction_suffix}_results.json` - Per-question steering effects
- `*_steering_{direction_suffix}_analysis.json` - Summary statistics
- `*_ablation_{direction_suffix}_results.json` - Ablation experiment results
- `*_ablation_{direction_suffix}_analysis.json` - Ablation statistics with p-values
- `*_{direction_suffix}_ablation_results.png` - Three-panel visualization
- `*_{direction_suffix}_ablation_summary.png` - Statistical summary

**Ablation statistical analysis:**

The ablation experiment tests whether the probe direction is *causally* involved in confidence judgments by removing it and measuring the effect on calibration. Statistical improvements include:

1. **Pooled null distribution:** Control effects are pooled across all layers (e.g., 7 layers × 20 controls = 140 samples) for more robust p-values
2. **FDR correction:** Benjamini-Hochberg correction for multiple layer testing
3. **Bootstrap CIs:** 95% confidence intervals on control effects
4. **Effect size (Z-score):** How many SDs the introspection effect is from the control mean

**Ablation visualization (three panels):**

1. **Absolute correlations:** Shows actual correlation values (baseline, introspection-ablated, control-ablated with SD). Negative correlation = well-calibrated.
2. **Differential effect with CI:** Bar chart of (introspection_Δcorr − control_Δcorr) with 95% CI. Colored by FDR significance.
3. **Distribution plot:** Violin plots showing control effect distribution per layer, with introspection effect overlaid. Shows where the effect falls in the null distribution.

### 6. Shared vs Unique Direction Transfer Analysis

**Script:** `analyze_shared_unique.py`

Tests whether the model uses a general or domain-specific uncertainty signal:
1. Loads MC directions from multiple datasets (e.g., SimpleMC, TriviaMC, GPQA)
2. Decomposes each direction into:
   - **Shared component:** Average of normalized MC directions (what's common)
   - **Unique component:** Residual (dataset-specific)
3. Tests whether probes along these directions transfer to meta activations

```bash
# Prerequisites: Run mc_entropy_probe.py on multiple datasets
python mc_entropy_probe.py --metric logit_gap  # SimpleMC
# (change DATASET_NAME and repeat for other datasets)

# Then analyze
python analyze_shared_unique.py --dataset SimpleMC
```

**Outputs:**
- `*_shared_unique_directions.npz` - Decomposed direction vectors
- `*_shared_unique_stats.json` - Decomposition statistics
- `*_{dataset}_shared_unique_transfer.json` - Transfer test results

### 7. Direction Analysis and Comparison

**Script:** `analyze_directions.py`

Analyzes and compares probe directions across experiments:
- Computes pairwise cosine similarities between direction types
- Runs logit lens analysis (projects directions through unembedding)
- Generates visualizations

```bash
python analyze_directions.py                    # Auto-detect directions
python analyze_directions.py --layer 15         # Focus on specific layer
```

### 8. Activation Patching

**Script:** `run_activation_patching.py`

Tests whether full activation patterns (not just 1D projections) are causally responsible for behavior. Unlike steering (which adds a scaled direction), patching replaces the entire activation vector from a source sample.

**Experiment types:**

1. **Cross-sample patching:** Replace activations from high-confidence samples into low-confidence samples (and vice versa). If activations encode confidence causally, this should swap behavior.

2. **Within-sample patching:** Compare patching with activations from the same sample vs. different samples as a control.

```bash
python run_activation_patching.py
```

**Configuration:**
```python
PATCHING_LAYERS = None           # Auto-select based on probe R², or specify list
SOURCE_SELECTION = "extreme"     # "extreme" (top/bottom quartile) or "random"
NUM_PATCHING_SAMPLES = 100       # Samples to patch
```

**Outputs:**
- `*_patching_results.json` - Per-sample patching effects
- `*_patching_analysis.json` - Summary statistics
- `*_patching_results.png` - Visualization

**Interpretation:** If patching high→low causes confidence to increase (and low→high causes it to decrease), this provides stronger causal evidence than correlation-based probing.

### 9. Direction Comparison (Clustering vs CAA vs Probes)

**Script:** `run_contrastive_direction.py` with `COMPARE_DIRECTIONS = True`

Compares different methods for finding uncertainty directions:

1. **Probe direction:** Linear regression on activations → metric
2. **CAA direction:** `mean(high_metric_activations) - mean(low_metric_activations)`
3. **Cluster directions:** Group activations by metric (quantile bins or k-means), compute centroids, use centroid-to-centroid vectors

**Why clustering?** Uncertainty might be encoded non-linearly or categorically (like one-hot vectors for "certain", "somewhat certain", "uncertain"). Clustering captures this by:
- Grouping activations into discrete states (low/mid/high metric values)
- Computing the centroid (mean activation) for each state
- Using the vector between centroids as the "direction"

**Clustering methods:**
- `"quantile"` - Group by metric percentiles (e.g., bottom/middle/top third)
- `"kmeans"` - Cluster in activation space, then label clusters by their mean metric

**What "works better" means:** Directions are compared by their causal effect on behavior—how much does steering along the direction change the model's confidence per unit multiplier? Larger effect = direction better captures the causal mechanism.

```bash
# Enable comparison mode in run_contrastive_direction.py:
COMPARE_DIRECTIONS = True
N_CLUSTERS = 3
CLUSTER_METHOD = "quantile"  # or "kmeans"

python run_contrastive_direction.py --metric logit_gap
```

**Outputs:**
- `*_direction_comparison.json` - Cosine similarities between direction types
- `*_direction_comparison.png` - Heatmap visualization
- `*_direction_steering_comparison.json` - Causal effect of each direction type

### 10. MC Answer Position Bias Analysis

**Script:** `analyze_mc_answer_bias.py`

Checks whether answer letter positions (A/B/C/D) correlate with uncertainty metrics. This helps interpret logit lens results—if the MC probe direction projects onto B/C-like tokens, is that because B/C answers actually correlate with uncertainty in the data?

```bash
# Configure at top of script:
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

python analyze_mc_answer_bias.py
```

**Outputs:**
- `*_mc_answer_bias.png` - Letter distribution and mean metric by letter
- `*_mc_answer_bias.json` - Spearman correlations between position and each metric

**Interpretation:** If there's no correlation between answer position and uncertainty metrics, the B/C pattern in logit lens is likely spurious or reflects model biases rather than dataset structure.

---

## Centralized Task Logic (`tasks.py`)

All prompt formatting and task-specific logic is centralized in `tasks.py`:

### Direct MC Task
```python
from tasks import MC_SETUP_PROMPT, format_direct_prompt

prompt, options = format_direct_prompt(question, tokenizer, use_chat_template=True)
```

### Stated Confidence Task
```python
from tasks import (
    STATED_CONFIDENCE_SETUP,
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
)

prompt, options = format_stated_confidence_prompt(question, tokenizer)
confidence = get_stated_confidence_signal(probs)  # Expected value over S-Z scale
```

### Answer or Delegate Task
```python
from tasks import (
    ANSWER_OR_DELEGATE_SETUP,
    ANSWER_OR_DELEGATE_SYSPROMPT,
    format_answer_or_delegate_prompt,
    get_answer_or_delegate_signal,
)

prompt, options, mapping = format_answer_or_delegate_prompt(question, tokenizer, trial_idx)
confidence = get_answer_or_delegate_signal(probs, mapping)  # P(Answer)
```

### Unified Response-to-Confidence
```python
from tasks import response_to_confidence

# Works for both task types
conf = response_to_confidence(response, probs, mapping, task_type="confidence")
conf = response_to_confidence(response, probs, mapping, task_type="delegate")
```

---

## Core Library (`core/`)

Reusable utilities for building experiments:

### `core/model_utils.py`
- `load_model_and_tokenizer()` - Load model with optional PEFT adapter and quantization
- `get_run_name()` - Generate consistent output filenames
- `is_base_model()`, `has_chat_template()` - Model property detection
- Supports `load_in_4bit` and `load_in_8bit` for large models

### `core/extraction.py`
- `BatchedExtractor` - Combined activation + logit extraction in single forward pass
- `compute_entropy_from_probs()` - Entropy computation

### `core/probes.py`
- `LinearProbe` - Ridge regression with optional PCA
- `train_and_evaluate_probe()` - Train and evaluate
- `permutation_test()` - Significance testing
- `run_layer_analysis()` - Full layer-by-layer analysis
- `compute_introspection_scores()` - Alignment metric
- `train_introspection_mapping_probe()` - Find mapping direction
- `compute_contrastive_direction()` - Alternative direction method
- `compute_cluster_centroids()` - Compute centroids for activation clusters (by metric quantiles or k-means)
- `compute_cluster_directions()` - Compute directions between cluster centroids (low→mid, mid→high, low→high)
- `compute_caa_direction()` - Compute Contrastive Activation Addition direction (mean high - mean low)
- `compare_directions()` - Compare multiple direction types by cosine similarity

### `core/steering.py`
- `SteeringHook` - Add direction to activations
- `AblationHook` - Remove projection onto direction
- `PatchingHook` - Replace full activation patterns (not just 1D projections)
- `BatchPatchingHook` - Batch version for patching multiple samples
- `steering_context()` / `ablation_context()` - Context managers
- `patching_context()` / `batch_patching_context()` - Context managers for activation patching
- `generate_orthogonal_directions()` - Control conditions

### `core/questions.py`
- `load_questions()` - Load with deterministic shuffling
- `format_direct_prompt()` - Thin wrapper around `tasks.format_direct_prompt()`
- `save_question_set()` / `load_question_set()` - Persist question sets
- `get_question_hash()` - Verify consistency across scripts

---

## Configuration

All scripts have configuration at the top:

```python
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned
DATASET_NAME = "SimpleMC"
NUM_QUESTIONS = 500
SEED = 42

# Meta-judgment task type
META_TASK = "confidence"  # or "delegate"

# Uncertainty metric (can also be set via --metric CLI flag)
METRIC = "logit_gap"  # entropy, top_prob, margin, logit_gap, top_logit
```

Output files automatically include model/dataset/metric info:
```
Llama-3.1-8B-Instruct_SimpleMC_introspection_logit_gap_results.json
```

When using a LoRA adapter:
```python
MODEL_NAME = "path/to/my-adapter"
# Outputs: Llama-3.1-8B-Instruct_adapter-my-adapter_SimpleMC_*
```

### Quantization for Large Models

For 70B+ models, use quantization:

```bash
python run_introspection_experiment.py --load-in-4bit --batch-size 2
python mc_entropy_probe.py --load-in-8bit --batch-size 4
```

---

## Typical Workflow

### Quick start (single metric)

```bash
# 1. Run introspection experiment
python run_introspection_experiment.py --metric logit_gap

# 2. Run steering with the probe directions
python run_introspection_steering.py --metric logit_gap
```

### Full workflow (comparing metrics)

```bash
# 1. Run introspection experiment (saves all metrics, probes one)
python run_introspection_experiment.py --metric logit_gap

# 2. Train introspection score probe
python run_introspection_probe.py --metric logit_gap

# 3. Run steering
python run_introspection_steering.py --metric logit_gap

# 4. Compare with entropy metric (re-uses saved activations)
python run_introspection_experiment.py --metric entropy  # Only re-trains probe
python run_introspection_probe.py --metric entropy
python run_introspection_steering.py --metric entropy
```

### Cross-dataset analysis

```bash
# Run MC probe on multiple datasets
python mc_entropy_probe.py --metric logit_gap  # DATASET_NAME = "SimpleMC"
# (change DATASET_NAME in script)
python mc_entropy_probe.py --metric logit_gap  # DATASET_NAME = "TriviaMC"
python mc_entropy_probe.py --metric logit_gap  # DATASET_NAME = "GPQA"

# Analyze shared vs unique directions
python analyze_shared_unique.py --dataset SimpleMC
```

---

## Key Concepts

**Introspection score:** `-metric_z × confidence_z`
- Positive when model is well-calibrated (high uncertainty + low confidence, or low uncertainty + high confidence)
- Negative when miscalibrated
- Sign is automatically handled for different metrics

**Direct→Meta transfer:** Train probe on direct MC activations, test on meta ("how confident?") activations. If it transfers, the model accesses similar representations when introspecting.

**Uncertainty probe vs. Introspection probe:**
- Uncertainty probe: direct activations → metric (where is uncertainty represented?)
- Introspection probe: meta activations → introspection_score (how does uncertainty become confidence?)

**Metrics comparison:**
- `entropy` is the classic information-theoretic measure
- `logit_gap` is often easier for linear probes (it's a linear function of the representation)
- Both capture similar information but may have different probe performance

---

## Other Files

### `load_and_format_datasets.py`
Dataset loading utilities (SimpleMC, GPQA, MMLU, TruthfulQA, etc.)

### `SimpleMC.jsonl`
Simple multiple-choice questions dataset.

### `run_contrastive_direction.py`
Alternative to regression: find direction via contrastive examples (well-calibrated vs miscalibrated). Also supports direction comparison mode (`COMPARE_DIRECTIONS = True`) for comparing probe, CAA, and cluster-based directions.

### `run_activation_patching.py`
Tests causal role of full activation patterns via cross-sample patching (replacing activations from high-confidence samples into low-confidence samples).

### `run_introspection_direction_experiment.py`
Focused analysis of the introspection mapping direction.
