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

## Workflow Overview

The experimental pipeline has three main phases:

1. **Train Probes** — Extract activations and train linear probes to find direction vectors
2. **Analyze Probes** — Interpret what the direction vectors represent
3. **Test Causality** — Verify directions are causally involved via steering/ablation/patching

---

## Phase 1: Train Probes

### 1.1 Next-Token Entropy Probe (General Case)

**Scripts:** `build_nexttoken_dataset.py` → `nexttoken_entropy_probe.py`

Tests whether next-token entropy (over the full vocabulary) is linearly decodable from activations on diverse text (Wikipedia, code, web crawl).

```bash
# Step 1: Build stratified dataset with diverse text
python build_nexttoken_dataset.py

# Step 2: Train entropy probes
python nexttoken_entropy_probe.py
python nexttoken_entropy_probe.py --metric logit_gap
python nexttoken_entropy_probe.py --all-metrics
```

**Outputs:** (prefixed with model name, e.g., `Llama-3.1-8B-Instruct_nexttoken_`)
- `*_samples_raw.json` - Raw samples with entropies (before stratification)
- `*_entropy_dataset.json` - Stratified dataset (even across entropy deciles)
- `*_activations.npz` - Extracted activations by layer
- `*_entropy_probe.json` - Probe R² by layer
- `*_entropy_probe.png` - Visualization

This is the most general test: can we decode entropy from activations on arbitrary text?

### 1.2 MC Uncertainty Probe (Factual MC Case)

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

### 1.3 Introspection Experiment (Direct→Meta Transfer)

**Script:** `run_introspection_experiment.py`

The core introspection test. For each question:
1. Runs direct MC prompt, extracts activations and uncertainty metrics
2. Runs meta prompt ("How confident are you?"), extracts activations and stated confidence
3. Trains probe: direct activations → selected metric
4. Tests transfer: does direct-trained probe work on meta activations?

```bash
python run_introspection_experiment.py --metric logit_gap
python run_introspection_experiment.py --metric entropy

# Optionally collect activations during other-confidence task for probe analysis
python run_introspection_experiment.py --metric entropy --collect-other-activations
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
- `*_{metric}_results.png` - Main visualization
- `*_{metric}_calibration_split.png` - Calibrated vs miscalibrated trials (side-by-side)
- `*_{metric}_other_confidence_probe.png` - Other-confidence probe comparison (if `--collect-other-activations`)

**Key metrics:**
- `direct_to_direct R²` - Can we predict the metric from direct activations? (sanity check)
- `direct_to_meta R²` - Does the probe transfer to meta activations? (introspection test)
- `behavioral correlation` - Does stated confidence correlate with actual uncertainty?

**Calibration split analysis:** Test set trials are automatically split into:
- **Calibrated trials:** Model's meta-judgment matches actual uncertainty (high confidence + low entropy, or low confidence + high entropy)
- **Miscalibrated trials:** Model's meta-judgment opposes actual uncertainty (high confidence + high entropy, or low confidence + low entropy)

Uses median split for balanced groups. D2D and D2M R² are computed separately for each subset to see whether transfer is driven by one subset. If D2M transfer only works on calibrated trials, the model might only be introspecting when behaviorally correct. If D2M is equal/stronger on miscalibrated trials, the probe picks up entropy regardless of behavioral accuracy.

**Other-confidence probe analysis (`--collect-other-activations`):** When enabled, the script also:
1. Collects activations during the other-confidence control task (asking model to estimate human difficulty instead of its own confidence)
2. Trains probes: other-confidence activations → direct entropy
3. Tests transfer: O→O (sanity check) and O→S (other-confidence → self-confidence meta)
4. Generates side-by-side comparison figure

This tests whether "other-confidence" activations also encode the model's own uncertainty. If O→S transfer is strong, it suggests the model encodes uncertainty similarly across different meta-judgment tasks, not specifically during self-confidence introspection. Adds ~2x inference time for the confidence task.

### 1.4 Contrastive Direction Computation

**Script:** `compute_contrastive_directions.py`

Computes direction vectors using contrastive examples rather than regression:

1. **Confidence direction:** `mean(high_confidence) - mean(low_confidence)` within well-calibrated examples
2. **Calibration direction:** `mean(calibrated) - mean(miscalibrated)` across all examples

```bash
python compute_contrastive_directions.py
```

**Configuration:**
```python
DATASETS = ["SimpleMC", "TriviaMC"]  # Process multiple datasets
META_TASKS = ["confidence", "delegate"]  # Process multiple task types
DIRECTION_TYPES = ["confidence", "calibration"]  # Which directions to compute
```

**Outputs:**
- `*_contrastive_confidence_directions.npz` - Confidence direction vectors
- `*_contrastive_calibration_directions.npz` - Calibration direction vectors
- `*_confidence_quality.png` - Direction quality visualization
- `*_calibration_quality.png` - Direction quality visualization

### 1.5 Introspection Score Probe (Meta→Calibration)

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

---

## Phase 2: Analyze Probes

### 2.1 Direction Analysis and Comparison

**Script:** `analyze_directions.py`

Analyzes and compares probe directions across experiments:
- Computes pairwise cosine similarities between direction types
- Runs logit lens analysis (projects directions through unembedding)
- Generates visualizations

```bash
python analyze_directions.py                    # Auto-detect directions
python analyze_directions.py --layer 15         # Focus on specific layer
```

**Outputs:**
- `*_direction_similarity.png` - Heatmap of direction similarities
- `*_logit_lens.png` - Top tokens per direction/layer

### 2.2 Activation Oracles (Direction Interpretation)

**Script:** `act_oracles.py`

Uses the AO (Activation Oracle) adapter to interpret what probe direction vectors represent in natural language. While logit lens shows what tokens a direction projects to, activation oracles ask the model directly what concept the direction encodes.

**Two-phase workflow:**

1. **Generation:** For each layer's probe direction, add the direction to the model's activations and ask interpretation questions. The model's responses reveal what concept the direction represents.

2. **Analysis:** Parse responses, compute quality scores, detect themes, and generate summary outputs.

**Interpretation questions:**
- Q1: "What concept does this vector represent?"
- Q2: "Is this vector related to confidence/doubt, certainty/uncertainty, introspection/self-reflection, or something else?" (multiple choice)
- Q3: "What type of mental state does this vector encode?"

```bash
# Phase 1: Generate interpretations (requires GPU, slow)
python act_oracles.py

# Phase 2: Analyze existing interpretations (fast, CPU-only)
python act_oracles.py --analyze outputs/Llama-3.1-8B-Instruct_SimpleMC_introspection_entropy_ao_interpretations.json
```

**Quality assessment:**
- **Entropy consistency ratio:** `max_entropy / mean_entropy` across generated tokens. Low ratio = consistent generation (good). High ratio = one spike then repetitive (garbage).
- **Quality score:** `1 / (1 + log(ratio/5))` converts ratio to 0-1 scale.
- **Distinctiveness:** How much better than random baseline directions.

**Theme detection:**
- **Embedding similarity:** Compare responses to concept descriptions (confidence, uncertainty, introspection).
- **Keyword matching:** Count domain-specific keywords in responses.
- **MC choice:** Extract selection from Q2 multiple-choice response.

**Outputs:**
- `*_ao_interpretations.json` - Raw responses with entropy metrics
- `*_ao_analysis.json` - Parsed analysis with layer summaries
- `*_ao_summary.txt` - Human-readable summary table
- `*_ao_visualization.png` - Concept heatmap and quality metrics

**Relation to logit lens:** Logit lens and activation oracles are complementary approaches to direction interpretation:
- Logit lens: Project direction through unembedding → top tokens (mechanistic view)
- Activation oracles: Add direction to activations → ask model what it represents (semantic view)

---

## Phase 3: Test Causality

### 3.1 Steering/Ablation with Probe Directions

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

### 3.2 Contrastive Direction Steering/Ablation

**Script:** `run_contrastive_direction.py`

Tests causality of contrastive directions via steering and ablation experiments:
- Loads introspection data and computes contrastive directions
- Runs steering experiments with varying multipliers
- Runs ablation experiments comparing to random controls

```bash
python run_contrastive_direction.py --metric logit_gap
```

**Direction comparison mode:** Set `COMPARE_DIRECTIONS = True` to compare different methods for finding uncertainty directions:

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

**Outputs:**
- `*_steering_results.json` - Steering effects by multiplier
- `*_ablation_analysis.json` - Ablation effects with statistics
- `*_direction_comparison.json` - Cosine similarities between direction types (if enabled)
- `*_direction_comparison.png` - Heatmap visualization (if enabled)

### 3.3 Activation Patching

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

### 3.4 MC Answer Probe Causality Experiment

**Script:** `run_mc_answer_ablation.py`

Tests whether the MC answer probe direction (which predicts A/B/C/D answer choice) is causally involved in introspection. This addresses the hypothesis that D2M transfer R² and behavioral correlation rise together because both depend on the model's answer representation.

**What it tests:** If we ablate the MC answer direction during the meta task:
1. Does the entropy probe D2M transfer R² decrease?
2. Does the behavioral correlation (stated confidence vs actual entropy) decrease?

If ablating the MC answer direction degrades both metrics, it suggests the answer representation is causally upstream of introspection.

```bash
# Run full ablation experiment
python run_mc_answer_ablation.py --metric entropy

# Just compute direction similarity (fast, no model loading)
python run_mc_answer_ablation.py --metric logit_gap --similarity-only
```

**Prerequisites:** Run `run_introspection_experiment.py` first to generate:
- MC answer directions (`*_mc_answer_directions.npz`)
- Entropy/metric directions (`*_{metric}_directions.npz`)
- Paired data with direct metrics (`*_paired_data.json`)

**Direction extraction:** The MC answer probe is a 4-class LogisticRegression. To get a single direction, we take the first principal component of the 4 class coefficient vectors, then project back to original activation space.

**Outputs:**
- `*_mc_answer_ablation_results.json` - Full results including:
  - Direction similarity (MC answer vs entropy probe at each layer)
  - Behavioral ablation effects (correlation change, p-values, z-scores)
- `*_mc_answer_ablation.png` - Two-panel visualization:
  - Panel 1: Correlation change per layer (MC ablation vs controls)
  - Panel 2: Direction similarity curve across layers

**Statistical approach:** Same as `run_introspection_steering.py`:
- 25 control directions per layer (random orthogonal)
- Pooled p-values across all layers
- FDR correction (Benjamini-Hochberg)
- Z-scores vs control distribution

**Note on logit lens:** To analyze what the MC answer direction represents, use `analyze_directions.py` which will automatically discover and process the saved `*_mc_answer_directions.npz` files.

---

## Miscellaneous Analysis Scripts

### Shared vs Unique Direction Analysis

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

### MC Answer Position Bias Analysis

**Script:** `analyze_mc_answer_bias.py`

Checks whether answer letter positions (A/B/C/D) correlate with uncertainty metrics. This helps interpret logit lens results—if the MC probe direction projects onto B/C-like tokens, is that because B/C answers actually correlate with uncertainty in the data?

```bash
python analyze_mc_answer_bias.py
```

**Outputs:**
- `*_mc_answer_bias.png` - Letter distribution and mean metric by letter
- `*_mc_answer_bias.json` - Spearman correlations between position and each metric

**Interpretation:** If there's no correlation between answer position and uncertainty metrics, the B/C pattern in logit lens is likely spurious or reflects model biases rather than dataset structure.

### Contrastive Ablation Visualization

**Script:** `visualize_contrastive_ablation.py`

Creates visualizations from existing steering and ablation results:
- Confidence change vs multiplier (steering curve)
- Alignment change by layer
- Summary statistics

```bash
python visualize_contrastive_ablation.py
```

### Introspection Direction Experiment

**Script:** `run_introspection_direction_experiment.py`

Focused analysis of the introspection mapping direction. Combines probe training with steering experiments in a single script for quick iteration.

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

## Typical Workflows

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

# 3. Analyze directions
python analyze_directions.py
python act_oracles.py

# 4. Run steering
python run_introspection_steering.py --metric logit_gap

# 5. Compare with entropy metric (re-uses saved activations)
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

### Contrastive direction workflow

```bash
# 1. Run introspection experiment to get activations
python run_introspection_experiment.py --metric entropy

# 2. Compute contrastive directions
python compute_contrastive_directions.py

# 3. Analyze directions
python analyze_directions.py

# 4. Test causality
python run_contrastive_direction.py --metric entropy
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

### `logres_helpers.py`
Helper functions for logistic regression analysis.

### `ao_with_tests.py`
Alternative activation oracle implementation with built-in test cases.
