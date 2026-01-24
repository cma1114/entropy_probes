# Entropy Probes

Investigating whether LLMs have internal representations of their own uncertainty, and whether they access these representations during meta-cognitive tasks like reporting confidence.

## Goal

Test the hypothesis that LLMs genuinely introspect on their uncertainty rather than just pattern-matching on surface features. The core workflow:

1. **Identify**: Find internal correlates of output uncertainty (entropy, logit_gap, etc.)
2. **Transfer**: Test whether these correlates appear during meta-judgment tasks (confidence reports, delegation decisions)
3. **Causality**: Verify the relationship is causal via ablation (and steering, future)
4. **Interpret**: Understand what the uncertainty direction represents (future)

## Workflow

### Step 0: Establish that uncertainty is encoded

Before testing transfer, verify that activations encode uncertainty at all.

**MC task** (multiple choice questions):
```
python identify_mc_correlate.py
```

**Next-token task** (diverse text):
```
python identify_nexttoken_correlate.py
```

Both scripts:
- Extract activations from all layers
- Compute uncertainty metrics (entropy, logit_gap, etc.)
- Find directions using two methods:
  - `probe`: Ridge regression to predict metric
  - `mean_diff`: Simple centroid difference (mean of high - mean of low)
- Compare methods and report R² per layer

### Step 1: Test meta-transfer

Test whether directions found on direct tasks transfer to meta-judgment:

```
python test_meta_transfer.py
```

This loads directions from Step 0 and tests how well they predict the original metric when applied to activations from a meta-task (confidence rating or delegation game).

### Step 2: Test causality via ablation

Test whether directions are causally necessary for meta-judgments:

```
python run_ablation_causality.py
```

This ablates directions during meta-task inference and measures if the correlation between stated confidence and actual uncertainty degrades. Control ablations (random orthogonal directions) establish a null distribution.

**Configuration**:
```python
INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_SimpleMC"
METRIC = "entropy"
META_TASK = "confidence"  # or "delegate"
NUM_QUESTIONS = 100
NUM_CONTROLS = 25  # random directions per layer for null distribution
```

**Key design choices**:
- Tests ALL layers (no pre-filtering by transfer R²—let ablation determine what matters)
- Tests BOTH probe and mean_diff methods in a single run
- Pooled null distribution + FDR correction for robust statistics

**Outputs**:
```
outputs/
├── {model}_{dataset}_ablation_{task}_{metric}_results.json
├── {model}_{dataset}_ablation_{task}_{metric}_probe.png
├── {model}_{dataset}_ablation_{task}_{metric}_mean_diff.png
└── {model}_{dataset}_ablation_{task}_{metric}_comparison.png
```

## Direction-Finding Methods

Two fundamentally different approaches:

| Method | How it works | Strengths |
|--------|--------------|-----------|
| `probe` | Ridge regression: find direction that best predicts target | Optimized for prediction accuracy |
| `mean_diff` | `mean(top 25%) - mean(bottom 25%)` | Simple, interpretable, robust |

Both are computed automatically and compared.

## Experiments

### identify_mc_correlate.py

Find internal correlates on MC question answering.

**Configuration** (top of script):
```python
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET = "SimpleMC"  # or "TriviaMC", "GPQA", etc.
METRICS = ["entropy", "logit_gap"]  # which metrics to probe
NUM_QUESTIONS = 500
SEED = 42
```

**Outputs**:
```
outputs/
├── {model}_{dataset}_mc_activations.npz         # Reusable activations (all layers)
├── {model}_{dataset}_mc_dataset.json            # Full question metadata + all metrics
├── {model}_{dataset}_mc_entropy_distribution.png
├── {model}_{dataset}_mc_{metric}_directions.npz # Directions per metric (for transfer)
├── {model}_{dataset}_mc_{metric}_results.json   # Stats per metric (R², MAE, bootstrap CIs)
└── {model}_{dataset}_mc_{metric}_results.png    # R² curves per method
```

### identify_nexttoken_correlate.py

Find internal correlates on general next-token prediction.

**Configuration**:
```python
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = None  # auto-detects from model name
METRICS = ["entropy", "logit_gap"]
```

**Requires**: Stratified dataset from `build_nexttoken_dataset.py`

**Outputs**:
```
outputs/
├── {model}_nexttoken_activations.npz         # Reusable activations (all layers)
├── {model}_nexttoken_dataset.json            # Full sample metadata + all metrics
├── {model}_nexttoken_entropy_distribution.png
├── {model}_nexttoken_{metric}_directions.npz # Directions per metric (for transfer)
├── {model}_nexttoken_{metric}_results.json   # Stats per metric (R², MAE, bootstrap CIs)
└── {model}_nexttoken_{metric}_results.png    # R² curves per method
```

### test_meta_transfer.py

Test transfer from direct task to meta-judgment.

**Configuration**:
```python
INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_SimpleMC"  # Loads _directions.npz + _dataset.json
META_TASK = "confidence"  # or "delegate"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
METRICS = ["entropy", "logit_gap"]
```

**Output**: `outputs/{model}_{dataset}_transfer_{meta_task}.npz`

Contains:
- Transfer R² for each metric × method × layer
- Behavioral correlation (stated confidence vs actual uncertainty)
- Comparison of which method transfers best

## Metrics

All computed automatically:

| Metric | Formula | Higher means | Linear? |
|--------|---------|--------------|---------|
| `entropy` | -Σ(p log p) | More uncertain | No |
| `top_prob` | max(p) | More confident | No |
| `margin` | p₁ - p₂ | More confident | No |
| `logit_gap` | z₁ - z₂ | More confident | Yes |
| `top_logit` | z₁ - mean(z) | More confident | Yes |

Linear metrics (logit_gap, top_logit) are generally better targets for linear probes.

## Core Library

The `core/` directory provides reusable utilities:

- `model_utils.py`: Model loading, quantization support
- `extraction.py`: Batched activation extraction
- `metrics.py`: Uncertainty metric computation
- `directions.py`: Direction finding (probe, mean_diff)
- `steering.py`: Activation intervention hooks

## Key Files

```
entropy_probes/
├── identify_mc_correlate.py          # Step 0: MC task
├── identify_nexttoken_correlate.py   # Step 0: Next-token task
├── test_meta_transfer.py             # Step 1: Transfer test
├── run_ablation_causality.py         # Step 2: Ablation causality test
├── core/
│   ├── metrics.py                    # Metric computation
│   ├── directions.py                 # Direction finding
│   ├── extraction.py                 # Activation extraction
│   ├── steering.py                   # Activation intervention hooks
│   ├── steering_experiments.py       # Ablation experiment utilities
│   └── model_utils.py                # Model loading
├── tasks.py                          # Prompt templates
├── filter_by_difficulty.py           # Filter by model correctness
├── outputs/                          # Results
└── data/                             # Question datasets (see below)
```

## Datasets

Source datasets in `data/`:

| Dataset | Questions | Description |
|---------|-----------|-------------|
| PopMC | 14,267 | Popular culture MC questions |
| SimpleMC | 500 | Simple factual MC questions |
| TriviaMC | 2,416 | Trivia MC questions |

Experiments sample `NUM_QUESTIONS` (default 500) from these sources.

### Difficulty Filtering

To create balanced correct/incorrect subsets for better signal variance:

```bash
python filter_by_difficulty.py
```

Creates `data/TriviaMC_difficulty_filtered.jsonl` with 250 correct + 250 incorrect questions. Then use normally:

```python
# identify_mc_correlate.py
DATASET = "TriviaMC_difficulty_filtered"
```

The filtered dataset plugs directly into the existing pipeline.

## Legacy Scripts

The original experiment scripts remain in the root directory for reference:
- `run_introspection_experiment.py` - Original monolithic experiment
- `mc_entropy_probe.py` - Original MC probe training
- `nexttoken_entropy_probe.py` - Original next-token probe training

These are superseded by the cleaner scripts but kept for backwards compatibility.
