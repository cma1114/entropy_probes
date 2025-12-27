# Introspection Experiments

This repository contains tools for studying whether language models can introspect on their own uncertainty. The core question: when a model chooses to answer or delegate/pass, or reports confidence, is it actually accessing internal representations of its own entropy?

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full introspection experiment (MC questions + meta-judgments)
python run_introspection_experiment.py
```

## The Five Analyses

### 1. Is entropy predictable from activations? (General case)

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

### 2. Is entropy predictable from activations? (Factual MC case)

**Script:** `mc_entropy_probe.py`

Tests whether entropy over MC answer options (A/B/C/D) is predictable from activations on factual questions. This is a narrower test than #1, but directly relevant to the meta-judgment experiments.

```bash
python mc_entropy_probe.py
```

**Outputs:** (prefixed with model and dataset, e.g., `Llama-3.1-8B-Instruct_SimpleMC_mc_`)
- `*_entropy_dataset.json` - Questions with MC entropy values
- `*_activations.npz` - Extracted activations by layer
- `*_entropy_probe.json` - Probe R² by layer
- `*_entropy_probe.png` - Visualization

### 3. Does the entropy probe transfer to meta-judgments? (Direct→Meta transfer)

**Script:** `run_introspection_experiment.py`

The core introspection test. For each question:
1. Runs direct MC prompt, extracts activations and entropy
2. Runs meta prompt ("How confident are you?"), extracts activations and stated confidence
3. Trains probe: direct activations → entropy
4. Tests transfer: does direct-trained probe work on meta activations?

```bash
python run_introspection_experiment.py
```

**Meta-judgment task types:** Configure `META_TASK` at the top of the script:

- `META_TASK = "confidence"` (default) - Explicit confidence rating on S-Z scale (<5% to >90%)
- `META_TASK = "delegate"` - Answer vs Delegate choice; confidence = P(Answer)

The delegate task asks "Would you answer this yourself or delegate to a teammate who answers 50% correctly?" This operationalizes confidence as the model's willingness to answer rather than an explicit self-report.

When using `META_TASK = "delegate"`, output files include `_delegate` suffix (e.g., `*_introspection_delegate_*`).

**Outputs:** (prefixed with model and dataset, e.g., `Llama-3.1-8B-Instruct_SimpleMC_introspection_`)
- `*_paired_data.json` - Questions, entropies, confidences
- `*_direct_activations.npz` - Direct prompt activations
- `*_meta_activations.npz` - Meta prompt activations
- `*_probe_results.json` - Probe performance by layer
- `*_entropy_directions.npz` - Entropy probe directions for steering (per layer)
- `*_results.png` - Visualization

**Key metrics:**
- `direct_to_direct R²` - Can we predict entropy from direct activations? (sanity check)
- `direct_to_meta R²` - Does the entropy probe transfer to meta activations? (introspection test)
- `behavioral correlation` - Does stated confidence correlate with actual entropy?

For focused transfer analysis with permutation testing, also see `run_introspection_probe.py`.

### 4. Steering/ablation with probe directions

**Script:** `run_introspection_steering.py`

Tests whether steering along a probe direction affects the model's confidence-entropy alignment:
- Loads probe directions (entropy or introspection score)
- Adds scaled direction to activations during meta prompts
- Measures change in alignment (does steering make confidence match entropy better?)
- Compares to random orthogonal control directions

**Configuration options:**

1. `DIRECTION_TYPE` - Which probe direction to use:
   - `"entropy"` (default) - Uses entropy probe directions from `run_introspection_experiment.py` (direct activations → entropy). Tests whether the direct→meta transfer direction is causal.
   - `"introspection"` - Uses introspection score directions from `run_introspection_probe.py` (meta activations → introspection_score). Tests whether the calibration direction is causal.

2. `META_TASK` - Which meta-judgment task to run steering on:
   - `"confidence"` (default) - Explicit confidence rating on S-Z scale
   - `"delegate"` - Answer vs Delegate choice; confidence = P(Answer)

   **Important:** `META_TASK` should match the setting used in `run_introspection_experiment.py` to load the correct data files.

```bash
# Option A: Confidence task with entropy directions
python run_introspection_experiment.py  # META_TASK = "confidence"
python run_introspection_steering.py    # META_TASK = "confidence", DIRECTION_TYPE = "entropy"

# Option B: Delegate task with entropy directions
# In run_introspection_experiment.py, set META_TASK = "delegate"
python run_introspection_experiment.py
# In run_introspection_steering.py, set META_TASK = "delegate", DIRECTION_TYPE = "entropy"
python run_introspection_steering.py
```

**Outputs:** (prefixed with model and dataset, e.g., `Llama-3.1-8B-Instruct_SimpleMC_introspection_`)
- `*_steering_entropy_results.json` - Per-question steering effects (entropy direction)
- `*_steering_entropy_analysis.json` - Summary statistics
- `*_steering_entropy_results.png` - Visualization
- `*_ablation_entropy_results.json` - Ablation experiment results
- `*_ablation_entropy_analysis.json` - Ablation summary

For introspection direction, outputs use `_steering_results.json` (no `_entropy` suffix).
For delegate task, outputs include `_delegate` in the prefix (e.g., `*_introspection_delegate_steering_entropy_results.json`).

### 5. Introspection mapping direction (entropy → judgment)

**Script:** `run_introspection_direction_experiment.py`

Finds the direction that maps entropy to stated confidence (the "introspection mapping"):
- Computes introspection score = -entropy_z × confidence_z
- Trains probe: meta activations → introspection score
- Extracts direction for steering/ablation

This is different from the entropy probe (#1-4). The entropy probe finds *where entropy is represented*. The mapping probe finds *the direction along which the model translates entropy into confidence judgments*.

```bash
python run_introspection_direction_experiment.py
```

**Alternative approach (contrastive):** `run_contrastive_direction.py`

Instead of regression, select well-calibrated examples (high confidence + low entropy, or low confidence + high entropy) and compute:
```
direction = mean(well_calibrated) - mean(miscalibrated)
```

```bash
python run_contrastive_direction.py
```

**Outputs:** (prefixed with model and dataset, e.g., `Llama-3.1-8B-Instruct_SimpleMC_contrastive_`)
- `*_results.json` - Statistics and layer analysis
- `*_directions.npz` - Direction vectors for each layer
- `*_results.png` - Visualization

Both approaches are also available in `core/probes.py`:
- `train_introspection_mapping_probe()` - Regression approach
- `compute_contrastive_direction()` - Contrastive approach

These give similar directions when the relationship is roughly linear (the regression approach is a softer version of the contrastive approach).

### 6. Direction Analysis and Comparison

**Script:** `analyze_directions.py`

Analyzes and compares probe directions across experiments:
- Loads all direction files for a model
- Computes pairwise cosine similarities between direction types
- Runs logit lens analysis (projects directions through unembedding to see what tokens they point toward)
- Generates visualizations

```bash
python analyze_directions.py                    # Auto-detect directions in outputs/
python analyze_directions.py --layer 15         # Focus on specific layer
```

**Outputs:**
- `*_direction_analysis.json` - Full analysis results
- `*_direction_similarity_layer{N}.png` - Similarity matrix at layer N
- `*_direction_similarity_across_layers.png` - How similarities evolve across layers
- `*_logit_lens_{source}_{name}.png` - Token heatmaps across layers

**Direction files saved by probe scripts:**

| Script | Output File | Direction Type |
|--------|-------------|----------------|
| `nexttoken_entropy_probe.py` | `*_nexttoken_entropy_directions.npz` | Next-token entropy |
| `mc_entropy_probe.py` | `*_mc_entropy_directions.npz` | MC answer entropy |
| `run_introspection_experiment.py` | `*_introspection_entropy_directions.npz` | Direct→entropy probe |
| `run_contrastive_direction.py` | `*_contrastive_directions.npz` | Contrastive (high/low conf) |

**Logit lens interpretation:**
For a direction `d` at layer L, the logit lens computes `d @ lm_head.weight.T` to see what tokens the direction "points toward." This helps interpret what the model is representing along that direction.

### 7. Shared vs Unique Direction Transfer Analysis

**Script:** `analyze_shared_unique.py`

Tests whether the model uses a general or domain-specific uncertainty signal when performing meta-judgment. This script:
1. Loads MC entropy directions from multiple datasets (e.g., SimpleMC, TriviaMC, GPQA)
2. Decomposes each direction into:
   - **Shared component:** Average of normalized MC directions (what's common across datasets)
   - **Unique component:** Residual after projecting out the shared component (dataset-specific)
3. Trains 1D probes along these directions on direct activations
4. Tests whether those probes transfer to meta activations

**Key question:** When the model judges its own confidence, is it using:
- The **shared** uncertainty signal? (general uncertainty representation)
- The **unique** dataset-specific signal? (domain-specific calibration)

**Prerequisites:**
```bash
# Step 1: Run mc_entropy_probe.py on at least 2 datasets
# (Edit DATASET_NAME in mc_entropy_probe.py for each)
python mc_entropy_probe.py  # SimpleMC
python mc_entropy_probe.py  # TriviaMC (change DATASET_NAME)
python mc_entropy_probe.py  # GPQA (change DATASET_NAME)

# Step 2: Run introspection experiment on the dataset you want to test
python run_introspection_experiment.py  # With matching DATASET_NAME
```

**Usage:**
```bash
python analyze_shared_unique.py --dataset SimpleMC
```

**Outputs:**
- `*_shared_unique_directions.npz` - Decomposed direction vectors (shared + unique per dataset)
- `*_shared_unique_stats.json` - Decomposition statistics (variance explained, pairwise similarities)
- `*_{dataset}_shared_unique_transfer.json` - Transfer test results for the specified dataset
- `*_{dataset}_shared_unique_transfer.png` - Visualization of transfer results

**Key metrics:**
- **Variance explained by shared:** How much of each MC direction is captured by the shared component
- **Direct R²:** How well does each direction predict entropy on direct activations
- **Meta R²:** How well does the probe transfer to meta activations (the introspection test)
- **Transfer ratio:** Meta R² / Direct R² (1.0 = perfect transfer)

---

## Core Library (`core/`)

Reusable utilities for building experiments:

### `core/model_utils.py`
- `load_model_and_tokenizer()` - Load model with optional PEFT adapter
- `get_run_name()` - Generate consistent output filenames
- `is_base_model()`, `has_chat_template()` - Model property detection

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

### `core/steering.py`
- `SteeringHook` - Add direction to activations
- `AblationHook` - Remove projection onto direction
- `steering_context()` / `ablation_context()` - Context managers
- `generate_orthogonal_directions()` - Control conditions

### `core/questions.py`
- `load_questions()` - Load with deterministic shuffling
- `save_question_set()` / `load_question_set()` - Persist question sets
- `get_question_hash()` - Verify consistency across scripts

---

## Meta-judgment Tasks (`meta_tasks.py`)

Configurable meta-judgment tasks:

### `stated_confidence`
"How confident are you that you know the answer?"
- Options: S-Z scale (<5% to >90%)
- Signal: Expected confidence (weighted by probs)

### `answer_or_delegate`
"Would you answer this yourself or delegate to a teammate?"
- Options: 1/2 (Answer/Delegate, alternating to control position bias)
- Signal: P(Answer)

```python
from meta_tasks import get_meta_task, format_stated_confidence_prompt

task = get_meta_task("stated_confidence")
prompt, options = task["format_prompt"](question, tokenizer)
signal = task["get_signal"](probs)
```

---

## Other Files

### `load_and_format_datasets.py`
Dataset loading utilities (SimpleMC, GPQA, MMLU, TruthfulQA, etc.)

### `SimpleMC.jsonl`
Simple multiple-choice questions dataset.

---

## Configuration

All scripts have configuration at the top:

```python
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned
DATASET_NAME = "SimpleMC"
NUM_QUESTIONS = 500
SEED = 42
```

Output files automatically include model/dataset info. When using a LoRA adapter, set `MODEL_NAME` to the adapter path:

```python
MODEL_NAME = "path/to/my-adapter"  # Outputs: Llama-3.1-8B-Instruct_adapter-my-adapter_SimpleMC_*
```

---

## Ensuring Question Consistency

All scripts should use the same questions. Two approaches:

### 1. Same seed, same loading
```python
questions = load_questions(DATASET_NAME, NUM_QUESTIONS)
random.seed(42)  # Immediately before shuffle!
random.shuffle(questions)
```

### 2. Save and reload (preferred)
```python
from core import save_question_set, load_question_set

# First script: save
save_question_set(questions, "questions_500.json",
                  metadata={"dataset": "SimpleMC", "seed": 42})

# Later scripts: load
questions = load_question_set("questions_500.json")
```

---

## Typical Workflow

### Quick start (entropy direction steering)

```bash
# 1. Run introspection experiment (extracts activations, trains probes, saves directions)
python run_introspection_experiment.py

# 2. Run steering/ablation with entropy directions
python run_introspection_steering.py  # Uses DIRECTION_TYPE = "entropy" by default
```

### Full workflow

1. **Run introspection experiment:**
   ```bash
   python run_introspection_experiment.py
   ```
   This outputs probe results, activations, AND entropy directions.

2. **Steering with entropy direction:**
   ```bash
   # Set DIRECTION_TYPE = "entropy" in run_introspection_steering.py
   python run_introspection_steering.py
   ```
   Tests if the entropy probe direction is causal for confidence judgments.

3. **(Optional) Train introspection score probe:**
   ```bash
   python run_introspection_probe.py
   ```
   Trains probe on meta activations → introspection_score with permutation tests.

4. **(Optional) Steering with introspection direction:**
   ```bash
   # Set DIRECTION_TYPE = "introspection" in run_introspection_steering.py
   python run_introspection_steering.py
   ```

5. **(Optional) Mapping direction analysis:**
   ```bash
   python run_introspection_direction_experiment.py
   # Or: python run_contrastive_direction.py
   ```

---

## Key Concepts

**Introspection score:** `-entropy_z × confidence_z`
- Positive when model is well-calibrated (high confidence + low entropy, or low confidence + high entropy)
- Negative when miscalibrated

**Direct→Meta transfer:** Train probe on direct MC activations, test on meta ("how confident?") activations. If it transfers, the model accesses similar representations when introspecting.

**Entropy probe vs. Mapping probe:**
- Entropy probe: activations → entropy (where is entropy represented?)
- Mapping probe: activations → introspection_score (how does entropy become confidence?)
