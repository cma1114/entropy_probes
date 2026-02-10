# Entropy Probes

Investigating whether LLMs have internal representations of their own uncertainty, and whether they access these representations during meta-cognitive tasks like reporting confidence.

## Goal

Test the hypothesis that LLMs genuinely introspect on their uncertainty rather than just pattern-matching on surface features. The core workflow:

1. **Identify**: Find internal correlates of output uncertainty (entropy, logit_gap, etc.)
2. **Transfer**: Test whether these correlates appear during meta-judgment tasks (confidence reports, delegation decisions)
   - D→M transfer: Does d_mc_uncertainty from the MC task predict uncertainty in meta-task activations?
   - M→D prediction: Can we find a direction in meta activations that predicts MC uncertainty? Is it the same as d_mc? (metamcuncert approach)
3. **Causality**: Verify the relationship is causal via ablation and steering
4. **Interpret**: Understand what the uncertainty direction represents (logit lens, direction similarity)

## Quick Start

Edit the `MODEL`, `DATASET`, and `META_TASK` constants at the top of each script, then run the stages in order:

```bash
# Stage 1: Find uncertainty + answer directions
python identify_mc_correlate.py

# Stage 2: Test transfer to meta-task + find confidence directions
python test_meta_transfer.py

# Stage 3: Causal tests
python run_ablation_causality.py
python run_steering_causality.py

# Stage 4: Interpret directions
python analyze_directions.py
python compare_direction_types.py

# Summary: Cross-stage consolidated view
python summarize_results.py
```

Each script prints comprehensive results to console and saves JSON + PNG outputs.

## Workflow

### Stage 1: Identify uncertainty directions

Find directions in activation space that correlate with output uncertainty.

**MC task** (multiple choice questions):
```bash
python identify_mc_correlate.py
```

Finds both uncertainty directions (probe + mean_diff methods) and optionally answer directions (A/B/C/D classification). 

**Next-token task** (diverse text):
```bash
python identify_nexttoken_correlate.py
```

Requires a stratified dataset from `build_nexttoken_dataset.py`. Finds uncertainty directions and optionally output token directions (which token the model selected). 

Both scripts extract activations, compute uncertainty metrics, find directions using two methods (probe and mean_diff), and report per-layer R² with bootstrap 95% CIs.

### Stage 2: Test meta-transfer

Test whether directions found on the direct task transfer to meta-judgments:

```bash
python test_meta_transfer.py
```

Loads directions from Stage 1 and tests how well they predict uncertainty from meta-task activations (confidence rating or delegation game). Reports:
- Behavioral correlation (stated confidence vs actual uncertainty) with bootstrap CIs
- D→D and D→M R² per layer with CIs
- Transfer ratio (best D→M / best D→D) with introspection strength interpretation
- Answer direction analysis

Optional direction-finding from meta-task activations:
- `FIND_CONFIDENCE_DIRECTIONS = True`: Find directions predicting stated confidence
- `FIND_MC_UNCERTAINTY_DIRECTIONS = True`: Find directions in meta-task activations that predict MC uncertainty (the "metamcuncert" directions)

The metamcuncert approach is a key introspection test: we find directions in the model's meta-task activations (when it's being asked about confidence) that predict its actual MC-task uncertainty. These can then be compared via cosine similarity to the original d_mc_uncertainty to test whether the same geometric representation is used in both contexts.

### Stage 3: Causal tests

**Ablation** (necessity):
```bash
python run_ablation_causality.py
```

Ablates directions during meta-task inference and measures if calibration degrades. Tests all layers with FDR correction and bootstrap CIs on effect sizes.

**Steering** (sufficiency):
```bash
python run_steering_causality.py
```

Steers along directions at graded magnitudes and measures dose-response. Significant slopes with correct sign indicate causal sufficiency.

**Activation patching** (full activation):
```bash
python run_activation_patching.py
```

Pairs questions by metric value in MC tasks (e.g., low-entropy source ↔ high-entropy target) and swaps complete activation vectors between them during meta task inference. At each tested layer, the source question's hidden state is replaced with the target's activations. If the source's behavior shifts toward the target's uncertainty profile, this demonstrates that full activations (not just a 1D direction) causally encode uncertainty.

Both ablation and steering test probe and mean_diff methods, and support `DIRECTION_TYPE`:
- `"uncertainty"`: MC uncertainty directions (d_mc_uncertainty from identify_mc_correlate.py)
- `"answer"`: MC answer directions (A/B/C/D from identify_mc_correlate.py)
- `"confidence"`: Stated confidence directions (from test_meta_transfer.py)
- `"metamcuncert"`: MC uncertainty directions found from meta activations (from test_meta_transfer.py with `FIND_MC_UNCERTAINTY_DIRECTIONS = True`)

**Cross-direction causality** (internal effects):
```bash
python run_cross_direction_causality.py
```

Tests whether ablating one direction type affects the activation projection onto other direction types. For example: does ablating answer directions reduce the projection onto uncertainty directions? This measures internal representational causality rather than behavioral output effects. Reports effect matrix with bootstrap CIs and FDR correction.

**Visualize single ablation effect:**
```bash
python plot_ablation_effect.py uncertainty confidence 14
```

Creates a two-panel figure showing the effect of ablating one direction on another:
- Panel 1: Raw projection trajectories across layers (baseline vs post-ablation)
- Panel 2: Delta with error bars (significance indicator - bars not crossing zero = significant)

Useful for examining specific ablation effects in detail after running cross-direction causality.

### Stage 4: Interpret

**Logit lens + direction analysis:**
```bash
python analyze_directions.py
```

Projects directions through the unembedding matrix to see what tokens they correspond to. Reports direction similarity across experiment types.

**Direction type comparison:**
```bash
python compare_direction_types.py
```

Compares uncertainty, answer, and confidence directions via cosine similarity. Reports bootstrap CIs on mean cosine.

### Cross-stage summary

```bash
python summarize_results.py
```

Reads all `*_results.json` files for a given model/dataset and produces a compact cross-stage console summary plus a unified `*_summary.json`.

## Direction-Finding Methods

Two fundamentally different approaches, with variants for continuous vs categorical targets:

### Continuous targets (uncertainty metrics, confidence)

| Method | How it works | Strengths |
|--------|--------------|-----------|
| `probe` | Ridge regression: find direction `w` minimizing `\|\|Xw - y\|\|²` | Optimized for prediction; uses all data points |
| `mean_diff` | `mean(top 25%) - mean(bottom 25%)` | Simple, interpretable; focuses on extremes |

### Categorical targets (answer prediction: A/B/C/D)

| Method | How it works | Strengths |
|--------|--------------|-----------|
| `probe` | Logistic regression classifier | Optimized for classification accuracy |
| `centroid` | Per-class centroids; classify by nearest centroid | Simple, interpretable; no optimization |

The `centroid` method computes the mean activation for each answer class (A, B, C, D) and classifies test samples by which centroid they're closest to. It's the categorical analogue of `mean_diff`—both are non-optimized methods based on group means.

### Interpretation

When comparing directions across tasks (e.g., "do confidence directions align with uncertainty directions?"):
- Comparing `probe↔probe` or `mean_diff↔mean_diff` is apples-to-apples (same method, different tasks)
- Comparing `probe↔mean_diff` for the *same* task tells you whether both methods found the same structure
- Low agreement between methods suggests the target doesn't have a single clean linear direction

Both methods are computed automatically and compared in every stage.

## Shared Parameters

These constants must match across scripts for consistent results:

| Parameter | Default | Used by |
|-----------|---------|---------|
| `SEED` | `42` | All scripts |
| `PROBE_ALPHA` | `1000.0` | identify_mc, identify_nexttoken, test_meta, test_cross_dataset |
| `PROBE_PCA_COMPONENTS` | `100` | identify_mc, identify_nexttoken, test_meta, test_cross_dataset |
| `TRAIN_SPLIT` | `0.8` | identify_mc, identify_nexttoken, test_meta, test_cross_dataset, ablation, steering |
| `MEAN_DIFF_QUANTILE` | `0.25` | identify_mc, identify_nexttoken, test_meta, test_cross_dataset |

These are marked with `# Must match across scripts` in each file's constant block.

## Output Naming Convention

All outputs use the prefix `{model_short}_{dataset}` where `model_short` is derived from the HuggingFace model name (e.g., `Llama-3.1-8B-Instruct`). When quantization is enabled, a suffix is appended (e.g., `Llama-3.1-8B-Instruct_4bit`).

For causality outputs, `{dir_suffix}` encodes the direction type: `uncertainty_{metric}` for uncertainty directions, or just the direction type name for others (e.g., `answer`, `confidence`, `metamcuncert`).

| Stage | File pattern | Description |
|-------|-------------|-------------|
| **Identify** | `*_mc_activations.npz` | Reusable activations (all layers) |
| | `*_mc_dataset.json` | Question metadata + all metrics |
| | `*_mc_{metric}_directions.npz` | Directions per metric (for transfer) |
| | `*_mc_{metric}_probes.joblib` | Trained probes per layer |
| | `*_mc_{metric}_results.json` | Per-layer R², MAE, bootstrap CIs |
| | `*_mc_answer_directions.npz` | Answer directions (classifier + centroid) |
| | `*_mc_answer_probes.joblib` | 4-class answer probes per layer |
| | `*_mc_answer_results.json` | Per-layer answer accuracy |
| | `*_mc_distributions.png` | Metric distributions (one row per metric) |
| | `*_mc_directions.png` | 4-panel directions summary |
| **Identify (next-token)** | `*_nexttoken_activations.npz` | Reusable activations (all layers) |
| | `*_nexttoken_dataset.json` | Sample metadata + metric values |
| | `*_nexttoken_{metric}_directions.npz` | Directions per metric |
| | `*_nexttoken_{metric}_results.json` | Per-layer R², CIs |
| | `*_nexttoken_{metric}_results.png` | R² curves per method |
| | `*_nexttoken_token_results.json` | Token prediction accuracy |
| **Meta-task** | `*_meta_{task}_activations.npz` | Meta-task activations |
| | `*_meta_{task}_results.json` | Transfer R², behavioral analysis |
| | `*_meta_{task}_results.png` | Transfer plots |
| | `*_meta_{task}_metaconfdir_directions.npz` | Confidence directions (meta → stated conf) |
| | `*_meta_{task}_metamcuncert_directions.npz` | MC uncertainty directions from meta (meta → MC unc) |
| | `*_meta_{task}_metamcuncert_results.json` | R², cosine similarity to d_mc |
| **Causality** | `*_ablation_{task}_{dir_suffix}_results.json` | Ablation effects + FDR |
| | `*_ablation_{task}_{dir_suffix}_results.png` | Ablation plots |
| | `*_steering_{task}_{dir_suffix}_results.json` | Steering dose-response |
| | `*_steering_{task}_{dir_suffix}_results.png` | Steering plots |
| | `*_cross_direction_{metric}_results.json` | Cross-direction effects |
| | `*_cross_direction_{metric}_results.png` | Cross-direction heatmaps |
| | `*_cross_direction_{metric}_ablation_effect.png` | Single ablation trajectory plot |
| **Interpret** | `*_direction_analysis.json` | Logit lens + similarity |
| | `*_direction_comparison.json` | Direction type comparison |
| | `*_direction_comparison.png` | Comparison plots |
| **Summary** | `*_summary.json` | Cross-stage aggregation |

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

## Project Layout

```
entropy_probes/
  # Stage 1: Identify directions
  identify_mc_correlate.py           # MC uncertainty + answer directions
  identify_nexttoken_correlate.py    # Next-token uncertainty directions

  # Stage 2: Transfer to meta-task
  test_meta_transfer.py              # D→M transfer + confidence directions
  test_cross_dataset_transfer.py     # Cross-dataset generalization

  # Stage 3: Causal tests
  run_ablation_causality.py          # Ablation (uncertainty/answer/confidence)
  run_steering_causality.py          # Steering dose-response
  run_activation_patching.py         # Full activation patching
  run_cross_direction_causality.py   # Cross-direction causal effects
  plot_ablation_effect.py            # Visualize single cross-direction ablation

  # Stage 4: Interpretation
  analyze_directions.py              # Logit lens + direction similarity
  compare_direction_types.py         # Uncertainty vs answer vs confidence
  act_oracles.py                     # Activation oracle interpretation
  ao_interpreter.py                  # AO library

  # Cross-stage
  summarize_results.py               # Cross-stage consolidation
  cross_predict_confidence.py        # Self vs other-confidence cross-prediction
  synthesize_causal_results.py       # Synthesize ablation + steering results

  # Utilities
  filter_by_difficulty.py            # Create balanced datasets
  build_nexttoken_dataset.py         # Stratified next-token dataset

  # Support modules
  tasks.py                           # Prompt templates
  load_and_format_datasets.py        # Dataset loaders

  # Library
  core/
    model_utils.py                   # Model loading, quantization
    extraction.py                    # Batched activation extraction
    metrics.py                       # Uncertainty metric computation
    directions.py                    # Uncertainty direction finding (probe, mean_diff)
    answer_directions.py             # MC answer direction finding
    confidence_directions.py         # Confidence direction finding
    probes.py                        # Probe training utilities
    questions.py                     # Question handling utilities
    steering.py                      # Activation intervention hooks
    steering_experiments.py          # Ablation/steering experiment utilities
    config_utils.py                  # get_config_dict() for JSON metadata
    plotting.py                      # Centralized visualization helpers

  # Archived
  archive/                           # 22 legacy/debug/one-off scripts
  archive/originals/                 # Pre-modification snapshots
```

## Visualization

All plotting uses centralized helpers from `core/plotting.py`. Colors, figure sizes, DPI, grid style, CI band opacity, and marker styles are defined once and used by all scripts. To change how plots look:

- **Colors**: Edit `METHOD_COLORS`, `DIRECTION_COLORS`, `TASK_COLORS`, `SIGNIFICANCE_COLORS`, `CONDITION_COLORS`
- **Figure sizes**: Edit `SINGLE_PANEL`, `TWO_PANEL_WIDE`, `THREE_PANEL_WIDE`, etc.
- **Styling**: Edit `DPI`, `GRID_ALPHA`, `CI_ALPHA`, `MARKER_SIZE`, `LINE_WIDTH`

Scripts compose figures using helpers like `plot_layer_metric()`, `mark_significant_layers()`, `save_figure()`, etc.

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

## Archive

Legacy, debug, and one-off scripts are in `archive/` with an `archive/README.md` explaining their provenance. Pre-modification snapshots of scripts changed during the consolidation are in `archive/originals/`.
