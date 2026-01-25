# Implementation Plan: Cross-Prediction and Answer D2M Analysis

## Summary

Two features to assess whether the model has genuinely distinct self-representations:

1. **Feature #1 (Answer D2M)**: Data exists but visualization is missing
2. **Feature #3 (Cross-Prediction)**: New script to test if self-confidence probes predict other-confidence equally well

---

## Feature #1: Answer D2M Transfer - DATA EXISTS, VISUALIZATION MISSING

**Status**: The computation exists (`test_meta_transfer.py:1661-1795`) and data is saved to JSON under `answer_transfer` key. But **no visualization** is generated.

**What exists**:
- `answer_transfer.d2d`: Per-layer accuracy on direct task test set
- `answer_transfer.d2m_by_position.{pos}.{layer}.centered/separate`: D2M accuracy per layer/position

**What's missing**: A plot showing answer D2M transfer like the uncertainty D2M plots (layer-wise accuracy, d2d vs d2m comparison).

**Action**: Add visualization function to `test_meta_transfer.py` that plots:
1. D2D answer accuracy by layer (sanity check)
2. D2M answer accuracy by layer (centered scaling)
3. Comparison line showing chance level (25%)

This will be added to the existing plotting section (after line 1834) as `plot_answer_transfer_results()`.

---

## Feature #3: Cross-Prediction Script

**Goal**: Test whether self-confidence probes are specific to self, or also predict other-confidence equally well.

The comparison is specifically between:
- `confidence` task = self-confidence ("how confident am I in my answer?")
- `other_confidence` task = other-confidence ("how hard would this be for a human?")

### Create: `cross_predict_confidence.py`

#### Configuration
```python
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC_difficulty_filtered"

TRAIN_SPLIT = 0.8
SEED = 42
N_BOOTSTRAP = 100
```

#### Algorithm

1. **Load probes and activations** for both tasks:
   - `{base}_confidence_confidence_probes.joblib` - self probes
   - `{base}_other_confidence_confidence_probes.joblib` - other probes
   - `{base}_transfer_confidence_activations.npz` - self activations
   - `{base}_transfer_other_confidence_activations.npz` - other activations

2. **Compute 2x2 transfer matrix** at each layer:
   | Train \ Test | self | other |
   |--------------|------|-------|
   | self         | self→self R² | self→other R² |
   | other        | other→self R² | other→other R² |

   Where:
   - `self→self`: self_probes applied to self_acts, predicting self_confidence
   - `self→other`: self_probes applied to other_acts, predicting other_confidence
   - `other→self`: other_probes applied to self_acts, predicting self_confidence
   - `other→other`: other_probes applied to other_acts, predicting other_confidence

3. **Use `apply_probe_centered()`** from `core/directions.py:590-630` for rigorous transfer test

4. **Compute specificity metrics**:
   - `self_specificity = self→self R² / self→other R²`
   - `other_specificity = other→other R² / other→self R²`
   - If >> 1: Evidence of distinct representations
   - Computed per-layer and summarized at best layer

5. **Bootstrap for CIs** (resample predictions, not refit)

#### Key Code Pattern
```python
from core.directions import apply_probe_centered
import joblib
import numpy as np

base = INPUT_BASE_NAME

# Load self-confidence probes (trained on confidence task)
self_probes = joblib.load(f"{base}_confidence_confidence_probes.joblib")["probes"]

# Load other-confidence probes (trained on other_confidence task)
other_probes = joblib.load(f"{base}_other_confidence_confidence_probes.joblib")["probes"]

# Load activations from both tasks
self_acts = np.load(f"{base}_transfer_confidence_activations.npz")
other_acts = np.load(f"{base}_transfer_other_confidence_activations.npz")

# Get confidence values (target)
self_confidence = self_acts["confidences"]  # or "stated_confidence"
other_confidence = other_acts["confidences"]

# Example: self→other cross-prediction
for layer in self_probes:
    # Apply self-trained probe to other-task activations
    X_other = other_acts[f"layer_{layer}_final"][test_idx]
    y_other = other_confidence[test_idx]

    result = apply_probe_centered(
        X_other, y_other,
        self_probes[layer]["scaler"],
        self_probes[layer]["pca"],
        self_probes[layer]["probe"]
    )
    self_to_other_r2 = result["r2"]
```

#### Outputs
- `{base}_confidence_cross_prediction.json`: Full metrics per layer
- `{base}_confidence_cross_prediction.png`: Multi-panel figure:
  1. Transfer matrix heatmap (2x2 at best layer)
  2. Layer-wise R²: self→self, self→other, other→other, other→self
  3. Specificity ratio by layer
  4. Interpretation summary

#### Interpretation Guide
```json
{
  "self_specificity": 2.3,
  "other_specificity": 1.8,
  "interpretation": "Self-confidence probe is 2.3x better at predicting self than other. Evidence for distinct representations."
}
```

---

## Files to Modify/Create

| File | Action |
|------|--------|
| `cross_predict_confidence.py` | **CREATE** - Main cross-prediction script |
| `test_meta_transfer.py` | **MODIFY** - Add `plot_answer_transfer_results()` function and call it |

## Key Dependencies

- `core/directions.py:apply_probe_centered()` - Rigorous transfer test
- Existing `.joblib` probe files from `identify_confidence_correlate.py`
- Existing `.npz` activation files from `test_meta_transfer.py`

---

## Verification

1. Run `python cross_predict_confidence.py` on remote machine
2. Check that self→self R² > self→other R² (if distinct)
3. Review specificity ratios and visualization for layer patterns

---

## Expected Outcome

If self vs other confidence are genuinely distinct:
- `self_specificity >> 1` (e.g., 2-3x)
- `other_specificity >> 1`
- Cross-task R² significantly lower than within-task R²

If they share the same representation:
- Specificity ratios ≈ 1
- Cross-prediction works equally well
