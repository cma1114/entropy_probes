# Introspection Experiments

This repository contains tools for studying whether language models can introspect on their own uncertainty. The core question: when a model reports confidence, is it actually accessing internal representations of its own entropy?

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

**Outputs:**
- `introspection_paired_data.json` - Questions, entropies, confidences
- `introspection_direct_activations.npz` - Direct prompt activations
- `introspection_meta_activations.npz` - Meta prompt activations
- `introspection_probe_results.json` - Probe performance by layer
- `introspection_results.png` - Visualization

**Key metrics:**
- `direct_to_direct R²` - Can we predict entropy from direct activations? (sanity check)
- `direct_to_meta R²` - Does the entropy probe transfer to meta activations? (introspection test)
- `behavioral correlation` - Does stated confidence correlate with actual entropy?

For focused transfer analysis with permutation testing, also see `run_introspection_probe.py`.

### 4. Steering/ablation with the entropy direction

**Script:** `run_introspection_steering.py`

Tests whether steering along the entropy direction affects the model's confidence-entropy alignment:
- Loads probe directions from `run_introspection_probe.py`
- Adds scaled direction to activations during meta prompts
- Measures change in alignment (does steering make confidence match entropy better?)
- Compares to random orthogonal control directions

```bash
# First, train probes and extract directions
python run_introspection_probe.py

# Then run steering experiment
python run_introspection_steering.py
```

**Outputs:**
- `introspection_steering_results.json` - Per-question steering effects
- `introspection_steering_analysis.json` - Summary statistics
- `introspection_steering_results.png` - Visualization

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

**Outputs:**
- `contrastive_direction_results.json` - Statistics and layer analysis
- `contrastive_directions.npz` - Direction vectors for each layer
- `contrastive_direction_results.png` - Visualization

Both approaches are also available in `core/probes.py`:
- `train_introspection_mapping_probe()` - Regression approach
- `compute_contrastive_direction()` - Contrastive approach

These give similar directions when the relationship is roughly linear (the regression approach is a softer version of the contrastive approach).

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

Output files automatically include model/dataset info via `get_run_name()`.

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

1. **Run introspection experiment:**
   ```bash
   python run_introspection_experiment.py
   ```

2. **Train focused probe:**
   ```bash
   python run_introspection_probe.py  # Get directions
   ```

3. **Steering experiment:**
   ```bash
   python run_introspection_steering.py  # Test intervention
   ```

4. **Mapping direction analysis:**
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
