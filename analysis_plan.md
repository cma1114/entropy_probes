╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Mechanistic Analysis of Model Confidence Calibration

 The Core Phenomenon

 Models can state confidence (S-Z scale) or decide to delegate in ways that correlate with their actual output uncertainty (entropy/logit gap over MC options).

 The Question: What internal computation produces this correlation?

 ---
 The Two Primary Hypotheses

 Hypothesis 1: "I Know / I Don't Know" Direction

 There exists a direction in activation space encoding epistemic state:
 - High projection → "I know this answer"
 - Low projection → "I'm uncertain / don't know"

 This direction would be the same across contexts:
 - Present in MC task when model is about to answer
 - Present in meta-task when model is judging its confidence
 - d_mc (trained to predict entropy/logit_gap) is a proxy for this direction

 Key insight: This is a holistic "knowledge state" representation, not just "which letter wins."

 Hypothesis 2: Answer Token Predominance

 The model considers the A/B/C/D options and checks if one predominates:
 - Strong A activation, weak B/C/D → confident
 - All four similar → uncertain

 Key distinction from H1: This is about the classifier confidence of the answer, not a separate "knowing" state. The model might:
 1. Examine ABCD activations
 2. Compute how peaked the distribution is
 3. Use that as confidence signal

 Critical observation: d_answer predicts which letter, but what matters for confidence is how confident that prediction is.

 These Are Not Mutually Exclusive

 Both mechanisms could exist and operate together. The questions are:

 Existence questions:
 1. Does a holistic "I know" direction exist? (Would appear at some position, possibly before options)
 2. Does ABCD predominance also contribute? (Would appear at some position, possibly after options)

 Relationship questions:
 3. If both exist, are they the same direction? (cosine_sim)
 4. Do they appear at the same position or different positions?
 5. Does one produce the other? (Mediation)
 6. Are they redundant or additive for predicting confidence?

 Possible Configurations

 | Configuration    | Description                                                  |
 |------------------|--------------------------------------------------------------|
 | H1 only          | Holistic "knowing" direction; ABCD is downstream consequence |
 | H2 only          | No "knowing" state until ABCD is computed                    |
 | H1 then H2       | "Knowing" appears first, influences ABCD predominance        |
 | H2 then H1       | ABCD computed first, produces "knowing" state                |
 | Both independent | Both exist, both contribute, neither produces the other      |
 | Same thing       | H1 and H2 are different descriptions of the same signal      |

 ---
 Where in the Meta-Task Token Stream?

 The meta-task prompt contains multiple regions:
 [System prompt about rating confidence]
 [Embedded question text]
 [Options: A, B, C, D with their text]
 [Task instruction: rate confidence / decide to delegate]
 [Output generation]

 Multiple token positions could be relevant:
 - Tokens within the embedded question
 - Tokens after the question (?, newlines)
 - Tokens at/after each option
 - Tokens after all options
 - Tokens in the task instruction
 - Final token (where output is generated)

 The Position Question is Empirical for Both Directions

 Where does d_mc appear?
 - Could be early (during question processing)
 - Could be mid (after options are processed)
 - Could be late (at output generation)
 - Could be multiple positions (present throughout)

 Where does d_answer appear?
 - Could be at option tokens (as each option is processed)
 - Could be after all options (once comparison is possible)
 - Could be at final position (just before outputting)
 - Could be multiple positions

 The tests:
 - Probe with d_mc at each position, measure R²
 - Probe with d_answer at each position, measure R² and classifier accuracy
 - Do they appear at the same positions or different positions?
 - If different, which comes first?

 The Relationship Question (No Presumed Ordering or Position)

 Possibility: H1 influences H2
 - d_mc appears at some position P1
 - This influences d_answer at some position P2 ≥ P1
 - Observable: Ablating d_mc at P1 changes d_answer at P2

 Possibility: H2 influences H1
 - d_answer appears at some position P1
 - This shapes d_mc at some position P2 ≥ P1
 - Observable: Ablating d_answer at P1 changes d_mc at P2

 Possibility: Independent contributions
 - Both appear (possibly at different positions)
 - Both contribute to confidence output
 - Observable: Ablating one doesn't affect the other, but both affect output

 Possibility: Same underlying signal
 - H1 and H2 are different descriptions of the same representation
 - Observable: High cosine similarity, same R² across positions, same causal effect

 ---
 The Key Experiments to Distinguish

 1. Is d_mc the same as d_answer_confidence?

 What is d_answer_confidence? Not a direction trained to predict which letter, but rather the entropy of the answer classifier's predictions. If the classifier says
  "90% A, 10% other" that's high confidence. If "25% each" that's low confidence.

 Test:
 - Extract d_answer classifier confidence (e.g., max probability or entropy of its softmax)
 - Compare: Does d_mc predict this confidence? Or are they independent?
 - Compute cosine similarity between d_mc and the direction that predicts answer_classifier_confidence

 Expected results:
 - If H1 ≈ H2: High similarity, both predict uncertainty
 - If H1 ≠ H2: Low similarity, each provides independent signal

 2. Where do signals first appear?

 Test:
 - Probe with d_mc at each available position in the prompt
 - Probe with d_answer classifier confidence at each position
 - Measure R² at each

 Expected results:
 - If H1 appears early: d_mc decodable before options are processed
 - If H2 appears early: d_answer_confidence decodable at or after options
 - If both independent: Each appears at its own position(s)
 - If same signal: Both appear at same positions with similar R²

 3. Which is causally necessary?

 Test at each position:
 - Ablate d_mc at position X, measure calibration
 - Ablate d_answer at position X, measure calibration
 - Compare effect sizes

 Key insight: Test ablation at multiple positions throughout the prompt. Where does each direction have causal effect? The positions where ablation matters most
 reveals where the signal is being used.

 4. Does one produce the other?

 Cross-direction mediation test:
 - Ablate d_mc at layer L, measure d_answer_confidence at layer L+k
 - Ablate d_answer at layer L, measure d_mc at layer L+k

 Expected results:
 - If d_mc → d_answer_confidence: Ablating d_mc reduces d_answer_confidence downstream
 - If d_answer_confidence → d_mc: Ablating d_answer reduces d_mc downstream
 - If independent: Neither affects the other

 ---
 How They Might Be Combined

 If both H1 and H2 contribute:

 Additive Model

 confidence_output = α × d_mc_projection + β × d_answer_confidence
 Both directions independently contribute. Ablating either partially degrades calibration.

 Redundant Model

 Either signal is sufficient:
 confidence = f(d_mc) if d_mc is informative else f(d_answer_confidence)
 Ablating one has modest effect because the other compensates.

 Sequential Model (direction unknown a priori)

 One influences the other (either direction possible):
 d_mc → d_answer → confidence
 d_answer → d_mc → confidence
 Observable: Ablating the upstream one affects the downstream one. Cross-direction ablation reveals which, if any, sequential relationship exists.

 Gating Model

 The confidence output is gated by question type:
 if question_is_factual: use d_answer_confidence
 else: use d_mc
 Effect of ablation depends on question type.

 ---
 Observable Signatures by Configuration

 Configuration: H1 exists (holistic "I know" direction)

 - d_mc decodable at some position(s) in the meta-task prompt
 - d_mc predicts final confidence output
 - Ablating d_mc at those positions degrades calibration

 Configuration: H2 exists (ABCD predominance contributes)

 - d_answer_confidence decodable at some position(s)
 - d_answer_confidence predicts final confidence output
 - Ablating d_answer at those positions degrades calibration

 Configuration: H1 influences H2

 - d_mc appears at earlier position than d_answer_confidence
 - Ablating d_mc changes d_answer_confidence downstream
 - Cross-direction effect: d_mc → d_answer

 Configuration: H2 influences H1

 - d_answer_confidence appears at earlier position than d_mc
 - Ablating d_answer changes d_mc downstream
 - Cross-direction effect: d_answer → d_mc

 Configuration: Both independent

 - Both decodable (possibly at different positions)
 - Neither ablation affects the other
 - Both independently predict confidence
 - Combined ablation has larger effect than either alone

 Configuration: Same underlying signal

 - d_mc ≈ d_answer_confidence (high cosine similarity)
 - Both appear at same positions with same R²
 - Ablating either has equivalent effect
 - One is just a different description of the other

 ---
 The Key Mechanistic Distinctions

 1. Introspection vs Pattern Matching

 Introspection: The model accesses an internal representation of its own uncertainty.
 Pattern Matching: The model recognizes surface features (question length, topic, hedging words in options) that correlate with difficulty.

 Why this matters: If it's pattern matching, the correlation is spurious—the model doesn't "know" it's uncertain, it just recognizes hard questions. True
 introspection requires accessing computation state.

 Distinguishing test: Present questions where surface features are misleading (easy-looking but actually hard, or vice versa). Does calibration survive?

 2. Transfer vs Re-computation

 Transfer: The uncertainty direction d_mc from the MC task is reactivated/used during meta-task.
 Re-computation: The model processes the embedded question fresh and computes uncertainty de novo.

 Why this matters: Transfer implies the model maintains a persistent representation. Re-computation implies the model must "re-derive" its uncertainty from the
 question content each time.

 Distinguishing test: Compare d_mc (trained on MC activations) to metamcuncert (trained on meta activations to predict MC uncertainty). High cosine similarity →
 transfer. Low similarity but both predict → re-computation.

 3. Direct vs Answer-Mediated

 Direct: Uncertainty signal → confidence output
 Answer-Mediated: Uncertainty → answer clarity → confidence

 The answer-mediated path suggests that confidence is a proxy for "how clear is my answer?" rather than "how uncertain is my distribution?"

 Distinguishing test: Ablate d_answer. If confidence calibration degrades, the answer representation mediates. If calibration survives, uncertainty acts directly.

 4. Early vs Late Position

 Early: Uncertainty is encoded at the question tokens (?, newline after options) during forward pass through the embedded question.
 Late: Uncertainty is only computed/aggregated at the final token when generating response.

 Why this matters: Early computation suggests the model processes the question and "holds" uncertainty in its residual stream. Late computation suggests
 just-in-time aggregation.

 Distinguishing test: Probe/intervene at different token positions. Where does uncertainty first appear? Where is intervention most effective?

 5. Linear vs Non-Linear Encoding

 Linear: Uncertainty is a direction in activation space (what probes find).
 Non-Linear: Uncertainty is encoded in norms, specific neuron activations, or complex feature interactions.

 Why this matters: If non-linear, steering along d_mc won't work (or won't work proportionally). The probe might be finding a correlated shadow, not the actual
 encoding.

 Distinguishing test: Does steering produce dose-response? Do non-linear features (e.g., activation norm) predict uncertainty independently?

 ---
 Additional Hypotheses to Consider

 H5: Surface Heuristics

 The model uses question features (topic, wording, option similarity) as a heuristic for difficulty. This happens to correlate with actual uncertainty because both
 depend on objective question difficulty.

 Test: "Other confidence" task asks model to estimate HUMAN difficulty. If other_confidence correlates with uncertainty as well as self_confidence, the model may be
  reading difficulty, not introspecting.

 H6: Entropy Neurons (cf. Stolfo et al. 2024)

 Specific neurons in MLP layers control output entropy. These same neurons might be read during meta-task to assess prior uncertainty.

 Test: Identify entropy neurons via causal methods. Ablate them during meta-task. Does calibration degrade more than ablating d_mc?

 Difference from d_mc: A direction is a linear combination across the full hidden space. Entropy neurons are specific units. They might be orthogonal to d_mc but
 still carry the signal.

 H7: Attention-Mediated Reading

 The confidence-generating token attends to earlier tokens where uncertainty is encoded. The mechanism is attention, not residual stream propagation.

 Test: Attention knockout—block attention from final position to question positions. Does calibration degrade?

 Test: Analyze attention patterns. What does the confidence token attend to?

 H8: Task Vector Activation

 The meta-task prompt activates a "confidence rating circuit" learned during training. This circuit includes a mapping from uncertainty to confidence.

 Test: Does adding task-framing language ("You are an expert calibrator") change calibration? Does the task vector itself contain uncertainty information?

 H9: RLHF-Induced Calibration

 Calibration is a byproduct of RLHF training toward "helpful and honest" responses. The model learned that appropriate uncertainty expression is rewarded.

 Test: Compare base model vs RLHF model. Is calibration absent before RLHF?

 ---
 A Framework for Testing These Hypotheses

 The Critical Observations to Make

 For each hypothesis, what observation would confirm vs refute it?

 | Hypothesis       | Confirmed by                                                              | Refuted by                                                       |
 |------------------|---------------------------------------------------------------------------|------------------------------------------------------------------|
 | Introspection    | Calibration survives when surface features mislead                        | Calibration fails on novel question formats                      |
 | Pattern Matching | other_confidence (human difficulty) correlates as well as self_confidence | self_confidence >> other_confidence correlation with uncertainty |
 | Transfer         | cosine_sim(d_mc, metamcuncert) > 0.7                                      | cosine_sim < 0.3 but both predict                                |
 | Re-computation   | Low cosine_sim but both directions work                                   | High cosine_sim                                                  |
 | Direct pathway   | Ablating d_answer doesn't hurt calibration                                | Ablating d_answer destroys calibration                           |
 | Answer-mediated  | Ablating d_answer hurts calibration                                       | d_answer ablation has no effect                                  |
 | Early position   | Uncertainty decodable before final position                               | Only decodable at final                                          |
 | Late position    | Intervention only effective at final                                      | Intervention at earlier positions is sufficient                  |
 | Linear encoding  | Steering produces proportional dose-response                              | Steering has no effect or threshold effect                       |
 | Non-linear       | Norm/neuron features predict independently                                | Linear direction captures all variance                           |

 What's Been Established vs What's Open

 Established:
 1. Linear probes can extract uncertainty from activations (R²~0.5-0.6)
 2. D→M transfer works (R² > 0 in meta context)
 3. Transfer is weaker than within-domain (suggests transformation or partial access)
 4. Ablating directions reduces calibration (causal necessity)
 5. Steering can shift confidence (causal sufficiency, at least partially)
 6. Cross-dataset directions have low cosine similarity but predict equally well ("direction degeneracy")

 Open:
 1. Transfer vs Re-computation: cosine_sim(d_mc, metamcuncert) not systematically reported
 2. Token position effects: experiments default to "final", earlier positions unexplored
 3. Answer mediation: single mediation test exists, full path analysis incomplete
 4. Entropy neurons: not tested in this codebase
 5. Attention mechanisms: not examined
 6. RLHF contribution: not tested (would require base model comparison)

 ---
 Empirical Finding: Layer-wise Separation of Uncertainty and Answer Signals

 Key methodological insight: For answer directions (d_mcanswer, d_metamcanswer), using |projection| (absolute value) rather than signed projection captures "answer predominance" - whether the model has ANY answer strongly activated, regardless of which letter. This operationalizes H2: a model with a dominant answer representation is confident; diffuse activation across A/B/C/D indicates uncertainty.

 Results (Llama-3.3-70B, TriviaMC_difficulty_filtered, delegate task, logit_gap metric):

 Layer-wise pattern for predicting delegation confidence (meta output):

 | Layer range | d_mcuncert R² | d_metamcanswer R² | Dominant signal |
 |-------------|---------------|-------------------|-----------------|
 | 0-19        | ~0            | 0.01-0.05         | Neither         |
 | 20-27       | 0.32-0.36     | 0.13-0.17         | d_mcuncert      |
 | 28-34       | 0.01-0.35     | 0.01-0.26         | Mixed           |
 | 35-44       | 0.41-0.54     | 0.02-0.31         | d_mcuncert      |
 | 45-79       | 0.27-0.46     | 0.49-0.79         | d_metamcanswer  |

 Unique contributions (leave-one-out ΔR²):
 - d_metamcanswer: mean ΔR² = 0.11, best ΔR² = 0.39 at L79
 - d_mcuncert: mean ΔR² = 0.05, best ΔR² = 0.28 at L29
 - d_mcanswer: mean ΔR² = 0.06, best ΔR² = 0.22 at L50
 - d_metamcuncert: mean ΔR² = 0.02, best ΔR² = 0.38 at L29

 Interpretation:
 1. The transferred MC uncertainty direction (d_mcuncert) becomes predictive in the 20s-30s layers
 2. The answer activation strength (d_metamcanswer with |projection|) takes over from the mid-40s onward
 3. Both provide substantial unique information (non-zero ΔR²), so they're not redundant
 4. This suggests a processing sequence: the model first accesses its uncertainty representation from the MC task (H1), then later the strength of answer activation (H2) becomes the dominant signal

 This supports the "both independent" configuration: H1 and H2 are distinct mechanisms that both contribute to meta-judgment confidence, appearing at different layers in the forward pass.

 Note: d_metamcanswer with |projection| reaching R² ~ 0.79 is comparable to d_metaconfdir (the target-defined direction), suggesting answer activation strength captures most of the variance in later layers.

 ---
 The Token Position Question in Depth

 The Setup

 The meta-task prompt contains many token positions. The final token can attend to ALL previous positions. So "uncertainty at final" could mean:
 1. Uncertainty computed de novo at the final position
 2. Uncertainty computed earlier and read via attention

 Key Methodological Issue: Ablation at Position X Affects All Downstream Positions

 If I ablate d_mc at position X, every subsequent position is affected (they attend to X). This confounds:
 - "X contains the signal" vs "X propagates the signal"

 Cleaner tests:
 - Patching: Swap activations at position X only, see if behavior changes
 - Attention knockout: Block attention from final to X, see if calibration degrades
 - Position-specific probing: Where is d_mc first decodable? Where is ablation most effective?

 Expected Patterns by Mechanism

 | Observation                               | Early computation | Late computation | Attention-mediated   |
 |-------------------------------------------|-------------------|------------------|----------------------|
 | d_mc decodable at early positions?        | Yes               | No               | Yes                  |
 | d_mc decodable at final?                  | Yes (propagated)  | Yes              | Maybe                |
 | Patching at early position transfers?     | Yes               | No               | Depends on attention |
 | Patching at final transfers?              | Maybe             | Yes              | Yes                  |
 | Attention knockout (final → early) hurts? | -                 | -                | Yes                  |

 Representation Evolution Across Positions

 Both d_mc and d_answer might transform as they propagate through positions. A probe trained at position P1 might not work at position P2.

 For d_mc: Does the uncertainty direction look the same at all positions?
 For d_answer: Does the answer direction look the same at all positions?

 Test: Train separate probes at each position. Compare directions via cosine similarity.
 - High similarity: Same representation throughout
 - Low similarity: Representation evolves; need position-specific directions

 Key question: Do d_mc and d_answer appear at the same positions or different positions?
 - If different: Which comes first? Does one causally affect the other?
 - If same: Are they measuring the same underlying representation?

 ---
 The Causal Structure Question

 Multiple possible pathways to confidence output:

 Direct paths:
 - d_mc → confidence (uncertainty directly determines confidence)
 - d_answer → confidence (answer clarity directly determines confidence)

 Mediated paths:
 - d_mc → d_answer → confidence (uncertainty shapes answer representation, which shapes confidence)
 - d_answer → d_mc → confidence (answer clarity shapes uncertainty representation, which shapes confidence)

 Parallel paths:
 - d_mc and d_answer both → confidence (independent contributions)

 What the Causal Structure Tests Look Like

 To determine actual structure:
 1. Ablate d_mc at layer L, measure d_answer at layer L+k: Does d_mc affect d_answer?
 2. Ablate d_answer at layer L, measure d_mc at layer L+k: Does d_answer affect d_mc?
 3. Ablate d_mc, measure confidence change
 4. Ablate d_answer, measure confidence change
 5. Ablate both, compare to sum of individual effects

 Existing Evidence

 The mediation test in run_ablation_causality.py measures:
 - Ablate d_answer → measure change in d_delegate projection

 This tests d_answer → confidence link. Not yet tested: d_mc → d_answer, d_answer → d_mc, or parallel contributions.

 ---
 Synthesis: What Would Each Scenario Look Like?

 Scenario 1: Pure Introspection via Direct Transfer

 - d_mc (from MC task) is reactivated in meta context
 - cosine_sim(d_mc, metamcuncert) ≈ 1.0
 - Ablating d_mc destroys calibration
 - Ablating d_answer has no effect
 - Token position doesn't matter (same direction works everywhere)

 Scenario 2: Answer-Mediated Introspection

 - Model reconstructs/retrieves answer in meta context
 - Answer clarity (classifier confidence) determines confidence
 - Ablating d_answer destroys calibration
 - Ablating d_mc may have indirect effect (via d_answer)
 - Token position matters: answer computed at options, used at final

 Scenario 3: Re-computation

 - Model processes embedded question fresh
 - metamcuncert ≠ d_mc (different direction, same target)
 - Ablating d_mc has no effect (wrong direction)
 - Ablating metamcuncert destroys calibration
 - Uncertainty appears at question tokens, not transferred from MC context

 Scenario 4: Surface Heuristics (Not Introspection)

 - other_confidence ≈ self_confidence (both predict uncertainty equally)
 - Calibration fails on novel question formats
 - Direction ablation has weak/inconsistent effects
 - Model is recognizing question difficulty, not accessing internal state

 Scenario 5: Multiple Redundant Mechanisms

 - Direction degeneracy: many directions access same signal
 - Ablating single direction has partial effect
 - Ablating multiple directions simultaneously has larger effect
 - Subspace ablation (remove rank-k projection) needed

 ---
 Key Open Questions

 1. Transfer vs Re-computation: What is cosine_sim(d_mc, metamcuncert)? (Existing code computes this but hasn't been systematically reported)
 2. Token position: Where is uncertainty first decodable? Where is intervention most effective? (Infrastructure exists, experiments use only final position)
 3. Answer mediation: Does the full path d_mc → d_answer → confidence hold? (Partial test exists, full path analysis missing)
 4. Introspection vs surface features: Does other_confidence (human difficulty estimate) correlate as well as self_confidence? (Both tasks exist, comparison not
 systematically done)
 5. Non-linear encoding: Do activation norms or specific neurons predict uncertainty independently of linear directions? (Not tested in current codebase)

 ---
 Prioritized Hypotheses to Test

 Most Discriminative Tests

 1. Transfer vs Re-computation (Critical)
   - Compute and report cosine_sim(d_mc, metamcuncert) per layer
   - If high: same representation, transfer hypothesis
   - If low but both work: re-computation hypothesis
 2. Introspection vs Surface Features (Critical)
   - Compare correlation(self_confidence, uncertainty) vs correlation(other_confidence, uncertainty)
   - If equal: model reads difficulty, not internal state
   - If self >> other: true introspection
 3. Token Position for Both Directions (Important)
   - Where does d_mc first appear in meta activations?
   - Where does d_answer first appear in meta activations?
   - Do they appear at the same positions or different?
   - Where does patching each direction transfer behavior?
   - Where does ablating each direction affect output?
 4. Causal Structure (Important)
 Possible paths (not presupposed):
   - d_mc → confidence (direct uncertainty effect)
   - d_answer → confidence (direct answer effect)
   - d_mc → d_answer → confidence (uncertainty influences answer clarity)
   - d_answer → d_mc → confidence (answer clarity influences uncertainty representation)
   - d_mc and d_answer → confidence independently (parallel paths)

 Test: Cross-direction ablation at different layers to determine actual causal structure
 5. Non-linear Encoding (Lower priority)
   - Do norms/neurons predict independently?
   - Does steering produce non-proportional effects?

 ---
 Implementation Plan: Mapping Analyses to Existing Infrastructure

 The codebase already has infrastructure for most analyses. Below is a step-by-step plan reusing existing code.

 Step 1: Transfer vs Re-computation — cosine_sim(d_mc, metamcuncert)

 Status: Infrastructure exists, just needs to be run and results extracted.

 Files involved:
 - test_meta_transfer.py — produces metamcuncert directions when FIND_MC_UNCERTAINTY_DIRECTIONS=True (line 139)
 - analyze_directions.py — computes pairwise cosine similarities between all direction files

 What exists:
 - analyze_directions.py:657 — compute_cosine_similarity() function
 - analyze_directions.py:1054 — plot_similarity_across_layers() plots cosine across layers
 - analyze_directions.py:1257-1339 — "SELF vs OTHER METAMCUNCERT" console output already compares self_confidence vs other_confidence metamcuncert directions

 What to do:
 1. Ensure test_meta_transfer.py has run with FIND_MC_UNCERTAINTY_DIRECTIONS=True for both confidence and other_confidence tasks
 2. Run analyze_directions.py with DATASET_FILTER set appropriately
 3. The output direction_analysis_direction_similarity_across_layers.png shows cosine(d_mc, metamcuncert) per layer
 4. JSON output direction_analysis_direction_analysis.json contains similarity_summary with mean/max cosine per pair

 Interpretation key:
 - cosine_sim(mc_logitgap, d_meta_mc_uncert_confidence) > 0.7 → Transfer hypothesis (same direction)
 - cosine_sim < 0.3 but both predict uncertainty → Re-computation (different direction, same target)

 ---
 Step 2: Introspection vs Surface Features — self_confidence vs other_confidence

 Status: Infrastructure exists. Results may already be present.

 Files involved:
 - tasks.py:166-232 — format_stated_confidence_prompt() for self-confidence
 - tasks.py:392-449 — format_other_confidence_prompt() for other-confidence
 - test_meta_transfer.py — supports META_TASK = "confidence" and META_TASK = "other_confidence"
 - analyze_directions.py:1257-1339 — "SELF vs OTHER METAMCUNCERT" comparison (already exists!)

 What exists:
 - The test_meta_transfer.py script can run both meta-tasks
 - Both produce *_meta_{task}_transfer_results_{pos}.json with transfer R² values
 - analyze_directions.py compares d_meta_mc_uncert_confidence vs d_meta_mc_uncert_other_confidence

 What to do:
 1. Run test_meta_transfer.py with META_TASK = "confidence" and META_TASK = "other_confidence"
 2. Compare transfer R² for both: If they're equal, model reads difficulty (surface cues). If self >> other, genuine introspection.
 3. Run analyze_directions.py — the "SELF vs OTHER METAMCUNCERT" section shows per-layer cosine similarity
 4. Check interpretation:
   - High cosine (~1) across layers → same direction → surface cues dominate
   - Low cosine (~0) → different directions → potential genuine introspection

 Key output files:
 - *_meta_confidence_transfer_results_final.json — self-confidence transfer R²
 - *_meta_other_confidence_transfer_results_final.json — other-confidence transfer R²
 - direction_analysis_direction_analysis.json — cosine(self, other) per layer

 ---
 Step 3: Token Position for Both Directions (d_mc AND d_answer)

 Status: Infrastructure exists but only "final" position is currently tested. Requires config change.

 Files involved:
 - test_meta_transfer.py:161 — PROBE_POSITIONS = ["final"] (needs expansion)
 - tasks.py:547-635 — find_mc_positions() finds: question_mark, question_newline, options_newline, final
 - core/answer_directions.py — find_answer_directions_from_meta() for d_answer in meta context

 What exists:
 - test_meta_transfer.py already supports multi-position extraction (lines 1500-1595)
 - Output files include position suffix: *_transfer_results_{pos}.json, *_confdir_directions_{pos}.npz
 - run_ablation_causality.py:720 uses find_mc_positions() for position-specific ablation

 What to do:
 1. Edit test_meta_transfer.py line 161:
 PROBE_POSITIONS = ["question_mark", "question_newline", "options_newline", "final"]
 2. Run for META_TASK = "delegate" (or confidence)
 3. This produces separate direction files per position:
   - *_meta_delegate_transfer_results_question_mark.json
   - *_meta_delegate_transfer_results_question_newline.json
   - *_meta_delegate_transfer_results_options_newline.json
   - *_meta_delegate_transfer_results_final.json
 4. Compare R² across positions for both d_mc transfer AND d_answer classifier accuracy

 New analysis needed: Plot R² vs position to show where each signal first appears. This could be added as a simple post-processing script that loads the
 per-position JSON files and plots.

 ---
 Step 4: Causal Structure — Cross-Direction Ablation

 Status: Infrastructure exists in run_cross_direction_causality.py.

 Files involved:
 - run_cross_direction_causality.py:96-101 — DIRECTION_TYPES = ["uncertainty", "answer", "confidence", "metamcuncert"]
 - core/steering_experiments.py:256-305 — BatchAblationHook with intervention_position="indexed"

 What exists:
 - The script ablates direction X at layer N and measures direction Y at layers M > N
 - Reports effect sizes (Cohen's d) and delta_mean
 - Saves *_cross_direction_{metric}_results.json with full effect matrix

 What to do:
 1. Run run_cross_direction_causality.py with all four direction types
 2. The output matrix shows:
   - ablate_uncertainty → measure_answer: Does d_mc affect d_answer downstream?
   - ablate_answer → measure_uncertainty: Does d_answer affect d_mc downstream?
   - ablate_uncertainty → measure_confidence: Direct d_mc → confidence path
   - ablate_answer → measure_confidence: Direct d_answer → confidence path

 Interpretation:
 - If ablate_answer → measure_uncertainty has large effect but not vice versa → answer precedes uncertainty
 - If ablate_uncertainty → measure_answer has large effect → uncertainty precedes answer
 - If both have effects → bidirectional or parallel
 - If neither → independent paths to confidence

 ---
 Step 5: d_answer_confidence — Classifier Confidence vs Holistic Uncertainty

 Status: Partially implemented. Needs small addition.

 Key insight from plan: d_answer predicts which letter, but confidence calibration depends on how confident that prediction is.

 What exists:
 - core/answer_directions.py has 4-way A/B/C/D classifiers producing per-class probabilities
 - Classifier confidence = max(softmax) or -entropy(softmax)

 What to add:
 A small analysis comparing:
 1. d_mc projection (holistic uncertainty direction)
 2. d_answer classifier confidence (entropy of 4-way softmax)

 If these are highly correlated and have similar causal effects, H1 and H2 may be measuring the same thing.

 Implementation:
 - Extract classifier probabilities from apply_answer_classifier_centered() in core/answer_directions.py
 - Compute classifier_confidence = 1 - entropy(probs) / log(4) (normalized)
 - Compare correlation(d_mc_projection, classifier_confidence)
 - Compare both to stated confidence

 ---
 Execution Order

 1. Step 1: Run analyze_directions.py (read-only analysis of existing outputs)
 2. Step 2: Check if both self/other confidence results exist; if not, run test_meta_transfer.py for both
 3. Step 3: Modify PROBE_POSITIONS in test_meta_transfer.py and re-run
 4. Step 4: Run run_cross_direction_causality.py with all direction types
 5. Step 5: Add classifier confidence extraction (small code addition)

 ---
 Summary Table: Analyses → Scripts

 | Analysis                   | Script                           | Config Changes          | Output                                    |
 |----------------------------|----------------------------------|-------------------------|-------------------------------------------|
 | cosine(d_mc, metamcuncert) | analyze_directions.py            | DATASET_FILTER          | direction_analysis.json, similarity plots |
 | self vs other confidence   | test_meta_transfer.py × 2        | META_TASK               | Transfer R² per task                      |
 | Token positions            | test_meta_transfer.py            | PROBE_POSITIONS         | Per-position results                      |
 | Cross-direction causality  | run_cross_direction_causality.py | DIRECTION_TYPES         | Effect matrix                             |
 | d_answer confidence        | core/answer_directions.py + new  | Extract softmax entropy | Correlation analysis                      |
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
