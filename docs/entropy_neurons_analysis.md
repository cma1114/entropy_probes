# Entropy Neurons and Frequency Neurons: Relation to Uncertainty Probing

Analysis of Stolfo et al., "Confidence Regulation Neurons in Language Models" (NeurIPS 2024, arXiv:2406.16254v2) and its implications for this project.

## Summary of Stolfo et al.

The paper identifies two types of neurons in the final MLP layer that regulate output confidence:

### Entropy Neurons

**Mechanism:** High weight norm but low composition with the unembedding matrix. They don't push toward any particular token. Instead, they add norm to the residual stream, which causes the final LayerNorm to rescale all logits by a smaller factor — effectively raising the temperature of the softmax.

**Why this works:** Softmax is not scale-invariant. Multiplying all logits by c is equivalent to setting temperature T = 1/c:
- c > 1 (scaling up) → sharper distribution → lower entropy
- c < 1 (scaling down) → flatter distribution → higher entropy

So a neuron that increases residual stream norm → LayerNorm divides by more → logits shrink → higher entropy. The neuron controls *how confident* without changing *which token wins*.

**Why the last layer:** The final MLP output goes directly to FinalLayerNorm → Unembed with no further layers to dilute the norm signal. An entropy neuron in layer 15 would have its norm contribution modified by layers 16–31.

**Why it exists:** Cross-entropy loss penalizes overconfident wrong predictions catastrophically (−log(p) → ∞), so the model needs a calibration mechanism. An entropy neuron is a learned "temperature knob" — essentially the model independently discovering temperature scaling (Guo et al., 2017) end-to-end during pretraining. Context-dependent logic in earlier layers feeds into this neuron (factual/familiar → low activation → confident; ambiguous/novel → high activation → uncertain).

### Frequency Neurons

**Mechanism:** High composition with the unembedding matrix. They directly push the output distribution toward or away from the unigram distribution (base rate token frequencies from training data).

**Why it exists:** When uncertain, a sensible fallback is the base rate — "predict what's statistically common." The strength of this prior mixing is context-dependent: suppressed in predictable contexts, active in ambiguous ones.

### Two Complementary Axes

| Scenario | Entropy neuron | Frequency neuron | Result |
|---|---|---|---|
| Confident prediction | low (sharp) | suppressed | Peaked on specific token |
| Uncertain, structured | high (flat) | suppressed | Spread across contextually plausible tokens |
| Uncertain, fallback | high (flat) | active | Spread toward common tokens |

Entropy neurons control the **temperature** (how spread out). Frequency neurons control the **shape** when spread out (toward unigram prior or contextually distributed). Together they implement a two-dimensional uncertainty representation: confidence level × uncertainty type.

## Implications for This Project

### 1. Norm vs. Direction

Our probing approach assumes uncertainty is encoded as a **direction** in activation space (linear probe finds a weight vector w such that w·x predicts entropy). Entropy neurons suggest uncertainty is also encoded as **norm** of the residual stream, which is partially but not fully capturable by a linear probe.

A linear probe *can* capture norm information if the norm correlates with a consistent direction (e.g., if the entropy neuron's output vector is consistent across examples). But ablating along a single direction might miss the mechanism if the neuron works by adding norm along varying directions.

### 2. Why Mean-Diff and Probe Find Different Directions

If the model's uncertainty representation has two axes (entropy neurons for temperature, frequency neurons for distribution shape), then:
- **Probe** (ridge regression on all data): Finds the single direction that best predicts the scalar uncertainty metric — likely a mixture of both signals
- **Mean-diff** (top vs. bottom quartile centroids): Captures whatever differs most between high- and low-uncertainty examples — may weight the axes differently depending on which dominates in the extremes

This could explain method disagreement: they're projecting a 2D signal onto a 1D direction from different angles.

### 3. Last-Layer Specificity

Entropy neurons are found in the **final layer**. Our probing results typically show strong uncertainty signal in late layers. But the causal mechanism is specifically in the last MLP — earlier-layer signal may reflect upstream computation that *feeds into* the entropy neuron rather than the entropy neuron's output itself.

This has implications for causal interventions:
- Ablating uncertainty directions at the last layer: might directly interfere with entropy neuron outputs
- Ablating at earlier layers: might disrupt the information flow *to* the entropy neuron
- The mechanism is different, and the causal effect could manifest differently

### 4. Meta-Task Transfer

If the model uses entropy neurons for calibrating MC answers, does it use the same mechanism when *reporting* confidence? The meta-task (stating confidence) produces a different kind of output (a confidence number, not an MC answer), so the final-layer entropy neuron mechanism may not apply the same way.

This could explain the transfer gap: the uncertainty *representation* transfers across tasks (it's computed in earlier layers), but the *actuator* (entropy neuron) is task-specific to the output distribution being produced.

### 5. Two Types of Uncertainty Information

The existence of both entropy and frequency neurons suggests probing for a single "uncertainty" signal may be incomplete. The model may separately encode:
- **Confidence level** (entropy neuron axis): "how sure am I?"
- **Uncertainty character** (frequency neuron axis): "is my uncertainty structured (between plausible alternatives) or unstructured (falling back to base rates)?"

A probe trained on entropy as the target captures the first axis well but may miss or conflate the second.

## Potential Extensions

1. **Norm analysis**: Check whether residual stream norm at the last layer correlates with our uncertainty metrics, independent of directional projections. If it does, the entropy neuron mechanism is active.

2. **Per-neuron analysis**: Identify which final-layer MLP neurons have the entropy neuron signature (high weight norm, low unembedding composition) in Llama-3.1-8B-Instruct and check if our probe directions align with their output vectors.

3. **Separate probing**: Train separate probes for entropy (temperature-like) and KL-divergence from unigram (frequency-like) to disentangle the two axes.
