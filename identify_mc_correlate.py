"""
Stage 1. Find MC uncertainty and answer directions from model activations during
multiple-choice question answering. Tests whether activations encode uncertainty
metrics (entropy, logit_gap, etc.) using both probe and mean_diff methods, and
optionally finds answer (A/B/C/D) directions.

Inputs:
    data/{dataset}.jsonl                               Raw MC questions
    data/{dataset}_difficulty_filtered.jsonl            (alternative) Filtered MC questions

Outputs:
    outputs/{model_dir}/results/{dataset}_mc_results.json              Consolidated results
    outputs/{model_dir}/results/{dataset}_mc_distributions.png         Metric distributions
    outputs/{model_dir}/results/{dataset}_mc_directions.png            4-panel summary
    outputs/{model_dir}/working/{dataset}_mc_activations.npz           Cached activations
    outputs/{model_dir}/working/{dataset}_mc_{metric}_directions.npz   Direction vectors
    outputs/{model_dir}/working/{dataset}_mc_{metric}_probes.joblib    Trained probes
    outputs/{model_dir}/working/{dataset}_mc_answer_directions.npz     Answer directions
    outputs/{model_dir}/working/{dataset}_mc_answer_probes.joblib      Answer probes
    outputs/{model_dir}/working/{dataset}_mc_run.log                   Detailed log

    where {model_dir} = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    e.g., "Llama-3.1-8B-Instruct" or "Llama-3.1-8B-Instruct_4bit_adapter-lora"

Shared parameters (must match across scripts):
    SEED, PROBE_ALPHA, PROBE_PCA_COMPONENTS, TRAIN_SPLIT, MEAN_DIFF_QUANTILE

Run after: (none -- this is the first stage)
    Optionally filter_by_difficulty.py to produce difficulty-filtered datasets.
"""

import random
import numpy as np
from pathlib import Path
import json
import torch
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    get_model_dir_name,
    should_use_chat_template,
    BatchedExtractor,
    compute_mc_metrics,
    find_directions,
    METRIC_INFO,
    setup_run_logger,
    print_run_header,
    print_key_findings,
    print_run_footer,
    format_r2_with_ci,
    format_best_layer,
)
from core.directions import probe_direction  # For saving probe objects
from core.config_utils import get_config_dict, get_output_path, find_output_file
from core.plotting import plot_metric_distributions, plot_directions_summary
from core.questions import load_questions
from core.answer_directions import (
    find_answer_directions_both_methods,
    encode_answers,
)
from tasks import format_direct_prompt

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: path to PEFT/LoRA adapter
DATASET = "TriviaMC_difficulty_filtered"
METRICS = ["logit_gap"]  # Which metrics to analyze
NUM_QUESTIONS = 500

# --- Quantization ---
LOAD_IN_4BIT = False  # Set True for 70B+ models
LOAD_IN_8BIT = False

# --- Experiment ---
SEED = 42                    # Must match across scripts
BATCH_SIZE = 8
N_BOOTSTRAP = 100            # Bootstrap iterations for confidence intervals

# --- Direction-finding (must match across scripts) ---
PROBE_ALPHA = 1000.0         # Must match across scripts
PROBE_PCA_COMPONENTS = 100   # Must match across scripts
TRAIN_SPLIT = 0.8            # Must match across scripts
MEAN_DIFF_QUANTILE = 0.25    # Must match across scripts

# --- Answer directions ---
FIND_ANSWER_DIRECTIONS = True   # Find answer (A/B/C/D) directions from MC activations
ANSWER_PCA_COMPONENTS = 256     # More components for 4-class classification

# --- Output ---
# Paths are constructed via get_output_path() which routes to outputs/results/ or outputs/working/

# =============================================================================
# MAIN
# =============================================================================


def main():
    # Model directory for organizing outputs
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET  # Model prefix now in directory, not filename

    # Setup logging
    logger = setup_run_logger(None, base_name, "mc", model_dir=model_dir)
    config = {
        "model": MODEL,
        "dataset": DATASET,
        "metrics": METRICS,
    }
    print_run_header("identify_mc_correlate.py", 1, "Find MC uncertainty directions", config)

    # Log full config to file
    logger.section("Configuration")
    logger.dict({
        "model": MODEL,
        "adapter": ADAPTER,
        "dataset": DATASET,
        "metrics": METRICS,
        "num_questions": NUM_QUESTIONS,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "n_bootstrap": N_BOOTSTRAP,
        "probe_alpha": PROBE_ALPHA,
        "probe_pca_components": PROBE_PCA_COMPONENTS,
        "train_split": TRAIN_SPLIT,
        "mean_diff_quantile": MEAN_DIFF_QUANTILE,
        "find_answer_directions": FIND_ANSWER_DIRECTIONS,
    })

    # Load model
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    logger.info(f"Model loaded: {num_layers} layers, chat_template={use_chat_template}")

    # Load questions
    questions = load_questions(DATASET, num_questions=NUM_QUESTIONS, seed=SEED)
    logger.info(f"Loaded {len(questions)} questions")

    random.seed(SEED)
    random.shuffle(questions)

    # Get option token IDs
    option_keys = list(questions[0]["options"].keys())
    option_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in option_keys]
    logger.info(f"Option tokens: {dict(zip(option_keys, option_token_ids))}")

    all_activations = {layer: [] for layer in range(num_layers)}
    all_probs = []
    all_logits = []
    all_predicted = []

    with BatchedExtractor(model, num_layers) as extractor:
        for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
            batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

            prompts = []
            for q in batch_questions:
                prompt, _ = format_direct_prompt(q, tokenizer, use_chat_template)
                prompts.append(prompt)

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            layer_acts_by_pos, probs, logits, _ = extractor.extract_batch(input_ids, attention_mask, option_token_ids)

            # extract_batch returns {pos_name: [per-item dicts]}; we only need "final"
            for item_acts in layer_acts_by_pos["final"]:
                for layer, act in item_acts.items():
                    all_activations[layer].append(act)

            for p, l in zip(probs, logits):
                all_probs.append(p)
                all_logits.append(l)
                all_predicted.append(option_keys[np.argmax(p)])

    # Stack activations
    activations_by_layer = {
        layer: np.stack(acts) for layer, acts in all_activations.items()
    }
    logger.info(f"Activation shape per layer: {activations_by_layer[0].shape}")

    # Compute ALL metrics (not just requested ones, for dataset file)
    all_probs_arr = np.array(all_probs)
    all_logits_arr = np.array(all_logits)
    all_metrics = compute_mc_metrics(all_probs_arr, all_logits_arr, metrics=None)  # All metrics

    # Build metadata for each question
    metadata = []
    correct_count = 0
    for i, q in enumerate(questions):
        predicted = all_predicted[i]
        is_correct = predicted == q["correct_answer"]
        if is_correct:
            correct_count += 1

        item = {
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "options": q["options"],
            "probabilities": all_probs[i].tolist(),
            "logits": all_logits[i].tolist(),
        }
        # Add all metric values
        for m_name, m_values in all_metrics.items():
            item[m_name] = float(m_values[i])
        metadata.append(item)

    accuracy = correct_count / len(questions)

    # Per-position accuracy breakdown
    position_correct = {}
    position_total = {}
    for item in metadata:
        ans = item["correct_answer"]
        position_total[ans] = position_total.get(ans, 0) + 1
        if item["is_correct"]:
            position_correct[ans] = position_correct.get(ans, 0) + 1

    # Log accuracy details
    logger.section("MC Accuracy")
    logger.info(f"Overall: {correct_count}/{len(questions)} ({accuracy:.1%})")
    for pos in sorted(position_total.keys()):
        pos_acc = position_correct.get(pos, 0) / position_total[pos]
        logger.info(f"  Position {pos}: {pos_acc:.1%}")

    # Log metric distributions
    logger.section("Metric Distributions")
    for metric, values in all_metrics.items():
        logger.info(f"{metric}: mean={values.mean():.3f}, std={values.std():.3f}, "
                   f"range=[{values.min():.3f}, {values.max():.3f}]")

    # Save activations file (binary cache - not consolidated)
    activations_path = get_output_path(f"{base_name}_mc_activations.npz", model_dir=model_dir)
    act_save = {f"layer_{i}": activations_by_layer[i] for i in range(num_layers)}
    for m_name, m_values in all_metrics.items():
        act_save[m_name] = m_values
    np.savez_compressed(activations_path, **act_save)

    # Build consolidated results JSON (will be saved at the end)
    consolidated_results = {
        "format_version": 2,
        "config": get_config_dict(
            model=MODEL,
            adapter=ADAPTER,
            dataset=DATASET,
            num_questions=len(questions),
            seed=SEED,
            train_split=TRAIN_SPLIT,
            probe_alpha=PROBE_ALPHA,
            pca_components=PROBE_PCA_COMPONENTS,
            n_bootstrap=N_BOOTSTRAP,
            mean_diff_quantile=MEAN_DIFF_QUANTILE,
            find_answer_directions=FIND_ANSWER_DIRECTIONS,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "dataset": {
            "stats": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": len(questions),
                "per_position_accuracy": {
                    pos: position_correct.get(pos, 0) / position_total[pos]
                    for pos in sorted(position_total.keys())
                },
            },
            "data": metadata,
        },
        "metrics": {},  # Will be populated below
    }
    # Add metric stats to dataset section
    for m_name, m_values in all_metrics.items():
        consolidated_results["dataset"]["stats"][f"{m_name}_mean"] = float(m_values.mean())
        consolidated_results["dataset"]["stats"][f"{m_name}_std"] = float(m_values.std())

    # Find directions for each metric
    metrics_to_analyze = {m: all_metrics[m] for m in METRICS}
    all_results = {}
    key_findings = {}  # For console summary

    for metric in METRICS:
        target_values = metrics_to_analyze[metric]
        logger.section(f"Direction Finding: {metric.upper()}")

        # Run parallel direction finding WITH bootstrap for R² confidence intervals
        results = find_directions(
            activations_by_layer,
            target_values,
            methods=["probe", "mean_diff"],
            probe_alpha=PROBE_ALPHA,
            probe_pca_components=PROBE_PCA_COMPONENTS,
            probe_n_bootstrap=N_BOOTSTRAP,
            probe_train_split=TRAIN_SPLIT,
            mean_diff_quantile=MEAN_DIFF_QUANTILE,
            seed=SEED,
            return_scaler=True,
        )

        all_results[metric] = results

        # Log per-layer results to file
        for method in ["probe", "mean_diff"]:
            logger.info(f"\n{method.upper()} per-layer R²:")
            fits = results["fits"][method]
            for layer in sorted(fits.keys()):
                r2 = fits[layer]["r2"]
                ci_lo = fits[layer].get("r2_ci_low")
                ci_hi = fits[layer].get("r2_ci_high")
                corr = fits[layer].get("corr", 0)
                ci_str = f" [{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None else ""
                logger.info(f"  Layer {layer:2d}: R²={r2:.4f}{ci_str}, r={corr:.3f}")

            # Find best layer for key findings
            best_layer = max(fits.keys(), key=lambda l: fits[l]["r2"])
            best_r2 = fits[best_layer]["r2"]
            ci_lo = fits[best_layer].get("r2_ci_low")
            ci_hi = fits[best_layer].get("r2_ci_high")
            key_findings[f"{metric}/{method}"] = format_best_layer(
                method, best_layer, best_r2, ci_lo, ci_hi
            )

        # Fit probes separately for transfer tests
        probe_objects = {}
        for layer in tqdm(range(num_layers), desc=f"Fitting {metric} probes", leave=False):
            X = activations_by_layer[layer]
            _, info = probe_direction(
                X, target_values,
                alpha=PROBE_ALPHA,
                pca_components=PROBE_PCA_COMPONENTS,
                bootstrap_splits=None,
                return_probe=True,
            )
            probe_objects[layer] = {
                "scaler": info["scaler"],
                "pca": info["pca"],
                "ridge": info["ridge"],
            }

        # Save directions (binary cache - not consolidated)
        directions_path = get_output_path(f"{base_name}_mc_{metric}_directions.npz", model_dir=model_dir)
        probes_path = get_output_path(f"{base_name}_mc_{metric}_probes.joblib", model_dir=model_dir)

        dir_save = {
            "_metadata_dataset": DATASET,
            "_metadata_model": MODEL,
            "_metadata_metric": metric,
        }
        probe_save = {
            "metadata": {"dataset": DATASET, "model": MODEL, "metric": metric},
            "probes": probe_objects,
        }

        for method in ["probe", "mean_diff"]:
            for layer in range(num_layers):
                dir_save[f"{method}_layer_{layer}"] = results["directions"][method][layer]
                if method == "probe" and "scaler_scale" in results["fits"][method][layer]:
                    dir_save[f"{method}_scaler_scale_{layer}"] = results["fits"][method][layer]["scaler_scale"]
                    dir_save[f"{method}_scaler_mean_{layer}"] = results["fits"][method][layer]["scaler_mean"]

        np.savez(directions_path, **dir_save)
        joblib.dump(probe_save, probes_path)

        # Add to consolidated results
        metric_results = {
            "metric_stats": {
                "mean": float(target_values.mean()),
                "std": float(target_values.std()),
                "min": float(target_values.min()),
                "max": float(target_values.max()),
                "variance": float(target_values.var()),
                "median": float(np.median(target_values)),
                "iqr": float(np.percentile(target_values, 75) - np.percentile(target_values, 25)),
            },
            "results": {},
        }
        for method in ["probe", "mean_diff"]:
            metric_results["results"][method] = {}
            for layer in range(num_layers):
                layer_info = {}
                for k, v in results["fits"][method][layer].items():
                    if isinstance(v, np.ndarray):
                        continue
                    if isinstance(v, np.floating):
                        layer_info[k] = float(v)
                    elif isinstance(v, np.integer):
                        layer_info[k] = int(v)
                    else:
                        layer_info[k] = v
                metric_results["results"][method][layer] = layer_info

        consolidated_results["metrics"][metric] = metric_results

    # Log diagnostic summary
    logger.section("Diagnostic Summary")
    for metric_name, results in all_results.items():
        layers = sorted(results["fits"]["probe"].keys())
        n_layers = len(layers)
        early_layers = layers[:n_layers // 4]
        late_layers = layers[3 * n_layers // 4:]

        for method in ["probe", "mean_diff"]:
            fits = results["fits"][method]
            early_r2 = np.mean([fits[l]["r2"] for l in early_layers])
            late_r2 = np.mean([fits[l]["r2"] for l in late_layers])
            r2_increase = late_r2 - early_r2
            logger.info(f"{metric_name}/{method}: early={early_r2:.3f}, late={late_r2:.3f}, delta={r2_increase:+.3f}")

    # =========================================================================
    # ANSWER DIRECTIONS
    # =========================================================================
    answer_results = None
    if FIND_ANSWER_DIRECTIONS:
        logger.section("Answer Directions")

        # Extract model answers from already-available metadata
        model_answers = [q["predicted_answer"] for q in metadata]
        answer_dist = dict(zip(*np.unique(model_answers, return_counts=True)))
        logger.info(f"Answer distribution: {answer_dist}")

        # Encode answers to integers
        encoded_answers, answer_mapping = encode_answers(model_answers)
        logger.info(f"Answer mapping: {answer_mapping}")

        # Create train/test split
        indices = np.arange(len(metadata))
        train_idx, test_idx = train_test_split(
            indices,
            train_size=TRAIN_SPLIT,
            random_state=SEED,
            shuffle=True
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
        logger.info(f"Train/test split: {len(train_idx)}/{len(test_idx)}")

        # Find answer directions using both methods
        answer_results = find_answer_directions_both_methods(
            activations_by_layer,
            encoded_answers,
            train_idx,
            test_idx,
            n_components=ANSWER_PCA_COMPONENTS,
            random_state=SEED,
            n_bootstrap=N_BOOTSTRAP,
            train_split=TRAIN_SPLIT,
        )

        # Log per-layer results
        for method in ["probe", "centroid"]:
            logger.info(f"\n{method.upper()} per-layer accuracy:")
            fits = answer_results["fits"][method]
            for layer in sorted(fits.keys()):
                acc = fits[layer]["test_accuracy"]
                std = fits[layer].get("test_accuracy_std", 0)
                logger.info(f"  Layer {layer:2d}: {acc:.1%} +/- {std:.1%}")

            # Add to key findings
            best_layer = max(fits.keys(), key=lambda l: fits[l]["test_accuracy"])
            best_acc = fits[best_layer]["test_accuracy"]
            key_findings[f"answer/{method}"] = f"{method}: {best_acc:.1%} at layer {best_layer} (chance=25%)"

        # Save directions (binary cache)
        answer_dir_path = get_output_path(f"{base_name}_mc_answer_directions.npz", model_dir=model_dir)
        dir_save = {
            "_metadata_input_base": f"{model_dir}/{base_name}",
            "_metadata_n_classes": len(answer_mapping),
            "_metadata_answer_mapping": json.dumps(answer_mapping),
        }
        for method in ["probe", "centroid"]:
            for layer in range(num_layers):
                dir_save[f"{method}_layer_{layer}"] = answer_results["directions"][method][layer]
        np.savez(answer_dir_path, **dir_save)

        # Save probe objects (binary cache)
        answer_probes_path = get_output_path(f"{base_name}_mc_answer_probes.joblib", model_dir=model_dir)
        probe_save = {
            "metadata": {
                "input_base": f"{model_dir}/{base_name}",
                "n_classes": len(answer_mapping),
                "answer_mapping": answer_mapping,
                "train_split": TRAIN_SPLIT,
                "n_pca_components": ANSWER_PCA_COMPONENTS,
                "seed": SEED,
            },
            "probes": answer_results["probes"],
        }
        joblib.dump(probe_save, answer_probes_path)

        # Add to consolidated results
        answer_section = {
            "stats": {
                "n_samples": len(metadata),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "answer_distribution": {k: int(v) for k, v in answer_dist.items()},
                "answer_mapping": answer_mapping,
            },
            "results": {},
            "comparison": {},
        }
        for method in ["probe", "centroid"]:
            answer_section["results"][method] = {}
            for layer in range(num_layers):
                layer_info = {}
                for k, v in answer_results["fits"][method][layer].items():
                    if isinstance(v, np.floating):
                        layer_info[k] = float(v)
                    elif isinstance(v, np.integer):
                        layer_info[k] = int(v)
                    else:
                        layer_info[k] = v
                answer_section["results"][method][layer] = layer_info
        for layer in range(num_layers):
            answer_section["comparison"][layer] = {
                "cosine_sim": float(answer_results["comparison"][layer]["cosine_sim"])
            }

        consolidated_results["answer"] = answer_section

    # =========================================================================
    # SAVE CONSOLIDATED JSON
    # =========================================================================
    results_path = get_output_path(f"{base_name}_mc_results.json", model_dir=model_dir)
    with open(results_path, "w") as f:
        json.dump(consolidated_results, f, indent=2)

    # =========================================================================
    # CONSOLIDATED PLOTS
    # =========================================================================
    metrics_to_plot = {m: all_metrics[m] for m in METRICS}
    metric_info_map = {m: METRIC_INFO[m] for m in METRICS}

    distributions_path = get_output_path(f"{base_name}_mc_distributions.png", model_dir=model_dir)
    plot_metric_distributions(
        metrics_to_plot,
        metadata,
        metric_info_map,
        distributions_path,
        title_prefix="MC",
    )

    directions_plot_path = get_output_path(f"{base_name}_mc_directions.png", model_dir=model_dir)
    plot_directions_summary(
        all_results,
        answer_results,
        metrics_to_plot,
        metadata,
        metric_info_map,
        directions_plot_path,
        title_prefix="MC",
    )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================
    # Add accuracy to key findings
    key_findings["MC Accuracy"] = f"{accuracy:.1%} ({correct_count}/{len(questions)})"

    print_key_findings(key_findings)

    # Build output file list
    output_files = [
        results_path,
        activations_path,
        distributions_path,
        directions_plot_path,
    ]
    for metric in METRICS:
        output_files.append(get_output_path(f"{base_name}_mc_{metric}_directions.npz", model_dir=model_dir))
        output_files.append(get_output_path(f"{base_name}_mc_{metric}_probes.joblib", model_dir=model_dir))
    if FIND_ANSWER_DIRECTIONS:
        output_files.append(get_output_path(f"{base_name}_mc_answer_directions.npz", model_dir=model_dir))
        output_files.append(get_output_path(f"{base_name}_mc_answer_probes.joblib", model_dir=model_dir))

    print_run_footer(output_files, logger.log_file)


if __name__ == "__main__":
    main()
