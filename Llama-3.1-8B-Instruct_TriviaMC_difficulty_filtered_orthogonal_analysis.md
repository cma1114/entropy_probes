⏺ What This Analysis Does

  This script tries to separate true introspection (model knowing its own confidence) from surface difficulty cues (model just recognizing a question "looks hard").

  Method:
  - d_self = direction predicting model's confidence about its OWN answer
  - d_other = direction predicting model's confidence about what ANOTHER model would answer
  - d_introspection = d_self with the shared part removed (unique to self)
  - d_surface = d_other with the shared part removed (unique to other)

  ---
  Key Results

  1. High Overlap: 84% cosine similarity

  The self and other directions are nearly identical. 71% of variance is shared — most of what the model uses to judge its own confidence is the same as what it uses to judge another model's confidence (i.e., surface difficulty cues).

  2. Predictive Power — THE MAIN FINDING

                        self-task    other-task
    d_self              0.861        0.848      ← predicts BOTH equally
    d_other             0.861        0.850      ← predicts BOTH equally
    d_introspection     0.777        -0.099     ← predicts ONLY self ✓
    d_surface           -0.076       0.819      ← predicts ONLY other ✓

  This is clean separation:
  - d_introspection predicts self-confidence (R²=0.78) but NOT other-confidence (R²≈0)
  - d_surface predicts other-confidence (R²=0.82) but NOT self-confidence (R²≈0)

  Interpretation: There IS a unique introspective component (~29% of variance) that specifically tracks "how confident am I about MY answer" — distinct from generic "this question looks hard."

  3. Causal Experiments

  Steering: The original directions (d_self, d_other) causally shift confidence in all 8 layers. But d_introspection only shows 1 FDR-significant layer — the unique introspective signal is weaker causally.

  Ablation: Nothing reaches FDR significance (possibly underpowered with only 100 controls). The raw Δcorr for d_self/d_other is large (~0.6-0.9) but the orthogonalized directions show tiny effects (~0.02-0.09).

  ---
  Bottom Line

  The model DOES have a unique introspective representation — about 29% of its confidence signal is specific to self-assessment, not just surface difficulty. This component has strong predictive power but weaker causal influence than the full (shared) confidence direction.