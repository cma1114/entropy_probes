"""
Cross-stage consolidation. Reads all *_results.json files for a given model/dataset
prefix and produces a compact console summary plus a unified JSON aggregating key
metrics from every stage. Does not replace per-script output.

Inputs:
    outputs/{base}_mc_{metric}_results.json                (Stage 1: identify)
    outputs/{base}_mc_answer_results.json                  (Stage 1: answer directions)
    outputs/{base}_meta_{task}_results.json                (Stage 2: meta-transfer)
    outputs/{base}_{task}_confidence_results.json          (Stage 2: confidence)
    outputs/{base}_ablation_{task}_{metric}_results.json   (Stage 3: ablation)
    outputs/{base}_steering_{task}_{metric}_results.json   (Stage 3: steering)
    outputs/{base}_direction_analysis.json                 (Stage 4: logit lens)
    outputs/{base}_direction_comparison.json               (Stage 4: direction types)

Outputs:
    outputs/{base}_summary.json    Unified cross-stage summary

Shared parameters (must match across scripts):
    LOAD_IN_4BIT, LOAD_IN_8BIT    Must match to produce correct model name in prefix

Run after: any/all stages
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config_utils import get_config_dict

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
# Must match the values used in other pipeline scripts.
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DATASET = "TriviaMC_difficulty_filtered"
ADAPTER = None

# --- Quantization ---
LOAD_IN_4BIT = False  # Must match across scripts
LOAD_IN_8BIT = False

# --- Meta-task ---
META_TASK = "delegate"  # Primary meta-task to summarize

# --- Metrics ---
PRIMARY_METRIC = "entropy"  # Primary uncertainty metric for headline numbers
ALL_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]

# --- Output ---
OUTPUT_DIR = Path("outputs")


# =============================================================================
# HELPERS
# =============================================================================

def _load_json(path: Path) -> Optional[Dict]:
    """Load a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_model_short() -> str:
    """Get short model name with quantization suffix."""
    from core import get_model_short_name
    return get_model_short_name(MODEL, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT)


def _get_base_name() -> str:
    """Get the {model_short}_{dataset} prefix."""
    model_short = _get_model_short()
    if ADAPTER:
        from core import get_model_short_name
        adapter_short = get_model_short_name(ADAPTER)
        return f"{model_short}_adapter-{adapter_short}_{DATASET}"
    return f"{model_short}_{DATASET}"


def _fmt_r2(r2: Optional[float], ci_lo: Optional[float] = None, ci_hi: Optional[float] = None) -> str:
    """Format R² with optional CI."""
    if r2 is None:
        return "N/A"
    s = f"{r2:.3f}"
    if ci_lo is not None and ci_hi is not None:
        s += f" [{ci_lo:.3f}, {ci_hi:.3f}]"
    return s


def _fmt_pct(val: Optional[float], ci_lo: Optional[float] = None, ci_hi: Optional[float] = None) -> str:
    """Format a percentage with optional CI."""
    if val is None:
        return "N/A"
    s = f"{val:.1%}"
    if ci_lo is not None and ci_hi is not None:
        s += f" [{ci_lo:.1%}, {ci_hi:.1%}]"
    return s


# =============================================================================
# STAGE PARSERS
# =============================================================================

def parse_identify_results(base_name: str) -> Dict[str, Any]:
    """Parse Stage 1 (identify) results."""
    identify = {"per_metric": {}}

    for metric in ALL_METRICS:
        path = OUTPUT_DIR / f"{base_name}_mc_{metric}_results.json"
        data = _load_json(path)
        if data is None:
            continue

        metric_info = {}

        # Extract metric distribution stats
        if "metric_stats" in data:
            metric_info["metric_stats"] = data["metric_stats"]

        # Extract per-method best results
        results = data.get("results", data)  # handle both nested and flat
        for method in ["probe", "mean_diff"]:
            method_data = results.get(method, {})
            if not method_data:
                continue

            # Find best layer by R²
            best_layer = None
            best_r2 = -float("inf")
            for layer_str, layer_data in method_data.items():
                try:
                    int(layer_str)  # ensure it's a layer number
                except (ValueError, TypeError):
                    continue
                r2 = layer_data.get("r2", -float("inf"))
                if r2 > best_r2:
                    best_r2 = r2
                    best_layer = int(layer_str)

            if best_layer is not None:
                layer_info = method_data[str(best_layer)]
                metric_info[method] = {
                    "best_layer": best_layer,
                    "best_r2": best_r2,
                    "best_r2_ci": [
                        layer_info.get("r2_ci_low"),
                        layer_info.get("r2_ci_high"),
                    ],
                    "best_pearson": layer_info.get("corr"),
                }
                # Fall back to std-based CI
                if metric_info[method]["best_r2_ci"][0] is None and "r2_std" in layer_info:
                    std = layer_info["r2_std"]
                    metric_info[method]["best_r2_ci"] = [
                        best_r2 - 1.96 * std,
                        best_r2 + 1.96 * std,
                    ]

        identify["per_metric"][metric] = metric_info

    # MC accuracy from dataset JSON
    dataset_path = OUTPUT_DIR / f"{base_name}_mc_dataset.json"
    dataset_data = _load_json(dataset_path)
    if dataset_data and "stats" in dataset_data:
        stats = dataset_data["stats"]
        identify["mc_accuracy"] = {
            "overall": stats.get("accuracy"),
            "n_questions": stats.get("num_questions"),
            "per_position": stats.get("per_position_accuracy", {}),
        }

    # Answer directions
    answer_path = OUTPUT_DIR / f"{base_name}_mc_answer_results.json"
    answer_data = _load_json(answer_path)
    if answer_data:
        identify["answer"] = {}
        for method in ["probe", "mean_diff"]:
            method_data = answer_data.get(method, {})
            if not method_data:
                continue
            best_layer = method_data.get("best_layer")
            best_acc = method_data.get("best_accuracy")
            if best_layer is not None:
                identify["answer"][method] = {
                    "best_layer": best_layer,
                    "best_accuracy": best_acc,
                    "best_accuracy_ci": [
                        method_data.get("best_accuracy_ci_low"),
                        method_data.get("best_accuracy_ci_high"),
                    ],
                }

    return identify


def parse_meta_results(base_name: str, task: str) -> Dict[str, Any]:
    """Parse Stage 2 (meta-transfer) results."""
    path = OUTPUT_DIR / f"{base_name}_meta_{task}_results.json"
    data = _load_json(path)
    if data is None:
        return {}

    meta = {}

    # Behavioral analysis
    behavioral = data.get("behavioral", {})
    if behavioral:
        meta["behavioral"] = {}
        meta["behavioral"]["token_choice"] = data.get("meta_target_stats", {})

        for metric in ALL_METRICS:
            if metric in behavioral:
                beh = behavioral[metric]
                r = beh.get("pearson_r")
                std = beh.get("pearson_r_std")
                meta["behavioral"].setdefault("correlations", {})[metric] = {
                    "pearson_r": r,
                    "pearson_ci": [
                        r - 1.96 * std if r is not None and std is not None else None,
                        r + 1.96 * std if r is not None and std is not None else None,
                    ],
                    "pearson_p": beh.get("pearson_p"),
                    "spearman_r": beh.get("spearman_r"),
                }

    # Transfer results (probe method)
    transfer_data = data.get("transfer", {})
    if transfer_data:
        meta["transfer"] = {}
        for method_key in ["probe", "mean_diff"]:
            # transfer section is keyed by metric, not method
            # The structure is: transfer[metric].d2m_centered.best_r2, transfer[metric].d2d.best_r2
            # Check if this data is nested under method or metric
            pass

        # Transfer is keyed by metric
        for metric in ALL_METRICS:
            metric_transfer = transfer_data.get(metric, {})
            if not metric_transfer:
                continue

            transfer_info = {}

            # D→D
            d2d = metric_transfer.get("d2d", {})
            if d2d:
                transfer_info["best_d2d"] = {
                    "layer": d2d.get("best_layer"),
                    "r2": d2d.get("best_r2"),
                }

            # D→M centered
            d2m_cen = metric_transfer.get("d2m_centered", {})
            if d2m_cen:
                transfer_info["best_d2m_centered"] = {
                    "layer": d2m_cen.get("best_layer"),
                    "r2": d2m_cen.get("best_r2"),
                    "r2_ci": [
                        d2m_cen.get("best_r2") - 1.96 * d2m_cen.get("best_r2_std", 0) if d2m_cen.get("best_r2_std") else None,
                        d2m_cen.get("best_r2") + 1.96 * d2m_cen.get("best_r2_std", 0) if d2m_cen.get("best_r2_std") else None,
                    ],
                    "pearson": d2m_cen.get("best_pearson"),
                }

            # D→M separate
            d2m_sep = metric_transfer.get("d2m_separate", {})
            if d2m_sep:
                transfer_info["best_d2m_separate"] = {
                    "layer": d2m_sep.get("best_layer"),
                    "r2": d2m_sep.get("best_r2"),
                }

            # Transfer ratio
            d2d_r2 = d2d.get("best_r2")
            d2m_r2 = d2m_cen.get("best_r2")
            if d2d_r2 and d2m_r2 and d2d_r2 > 0:
                transfer_info["transfer_ratio"] = d2m_r2 / d2d_r2

            meta["transfer"].setdefault(metric, {}).update(transfer_info)

    # Answer confound
    answer_transfer = data.get("answer_transfer", {})
    if answer_transfer:
        summary = answer_transfer.get("summary", {})
        meta["answer_confound"] = {
            "d2d_accuracy": summary.get("d2d_best_accuracy"),
            "d2m_accuracy": summary.get("d2m_best_accuracy"),
        }

    # Confidence directions
    conf_path = OUTPUT_DIR / f"{base_name}_{task}_confidence_results.json"
    conf_data = _load_json(conf_path)
    if conf_data:
        meta["confidence_directions"] = {}
        for method in ["probe", "mean_diff"]:
            method_data = conf_data.get(method, {})
            if method_data:
                meta["confidence_directions"][method] = {
                    "best_layer": method_data.get("best_layer"),
                    "best_r2": method_data.get("best_r2"),
                }

    return meta


def parse_causality_results(base_name: str, task: str) -> Dict[str, Any]:
    """Parse Stage 3 (causality) results — ablation and steering."""
    causality = {}
    model_short = _get_model_short()

    for stage_type in ["ablation", "steering"]:
        for metric in ALL_METRICS:
            # Try both naming patterns for the base
            candidates = [
                OUTPUT_DIR / f"{base_name}_{stage_type}_{task}_{metric}_results.json",
                OUTPUT_DIR / f"{model_short}_{DATASET}_{stage_type}_{task}_{metric}_results.json",
            ]
            # Also try with just the last part of dataset name (legacy pattern)
            dataset_parts = DATASET.split("_")
            if len(dataset_parts) > 1:
                candidates.append(
                    OUTPUT_DIR / f"{model_short}_{dataset_parts[0]}_{stage_type}_{task}_{metric}_results.json"
                )

            data = None
            for path in candidates:
                data = _load_json(path)
                if data is not None:
                    break

            if data is None:
                continue

            stage_info = {}

            # Extract comparison summary
            comparison = data.get("comparison", {})
            for method in ["probe", "mean_diff"]:
                method_comp = comparison.get(method, {})
                if not method_comp:
                    continue

                if stage_type == "ablation":
                    stage_info[method] = {
                        "n_significant": method_comp.get("n_significant_fdr005", method_comp.get("n_significant_bootstrap", 0)),
                        "n_layers_tested": method_comp.get("n_layers_tested"),
                        "best_layer": method_comp.get("best_bootstrap_layer", method_comp.get("best_layer")),
                        "best_effect": method_comp.get("best_bootstrap_effect"),
                        "best_effect_z": method_comp.get("best_effect_z"),
                    }
                else:  # steering
                    stage_info[method] = {
                        "n_significant": method_comp.get("n_significant_fdr005", method_comp.get("n_significant_fdr", 0)),
                        "n_sign_correct": method_comp.get("n_sign_correct_fdr005", method_comp.get("n_sign_correct_fdr", 0)),
                        "n_layers_tested": method_comp.get("n_layers_tested"),
                        "best_layer": method_comp.get("best_layer"),
                        "best_slope": method_comp.get("best_slope"),
                        "best_effect_z": method_comp.get("best_effect_z"),
                    }

            if stage_info:
                causality.setdefault(stage_type, {}).setdefault(task, {})[metric] = stage_info

    return causality


def parse_interpret_results(base_name: str) -> Dict[str, Any]:
    """Parse Stage 4 (interpretation) results."""
    interpret = {}

    # Direction analysis (logit lens)
    analysis_path = OUTPUT_DIR / f"{base_name}_direction_analysis.json"
    # Also check without dataset (analyze_directions uses model-only prefix)
    model_short = _get_model_short()
    analysis_path_alt = OUTPUT_DIR / f"{model_short}_direction_analysis.json"

    data = _load_json(analysis_path)
    if data is None:
        data = _load_json(analysis_path_alt)

    if data is not None:
        # Similarity summary
        sim_summary = data.get("similarity_summary", {})
        if sim_summary:
            interpret["direction_similarity"] = {}
            for pair_key, pair_data in sim_summary.items():
                interpret["direction_similarity"][pair_key] = {
                    "mean_abs_cosine": pair_data.get("mean_abs_cosine"),
                    "max_abs_cosine": pair_data.get("max_abs_cosine"),
                    "max_abs_layer": pair_data.get("max_abs_layer"),
                }

        # Logit lens top tokens for a few key layers
        per_layer = data.get("per_layer", {})
        if per_layer:
            logit_lens = {}
            for layer_str, layer_data in per_layer.items():
                lens_data = layer_data.get("logit_lens", {})
                if lens_data:
                    for dir_name, tokens_data in lens_data.items():
                        top_tokens = tokens_data.get("tokens", [])[:5]
                        logit_lens.setdefault(dir_name, {})[layer_str] = top_tokens
            if logit_lens:
                interpret["logit_lens"] = logit_lens

    # Direction comparison
    comp_path = OUTPUT_DIR / f"{base_name}_direction_comparison.json"
    comp_data = _load_json(comp_path)
    if comp_data:
        interpretations = comp_data.get("interpretations", {})
        if interpretations:
            interpret["direction_type_interpretations"] = interpretations

        comparisons = comp_data.get("comparisons", {})
        if comparisons:
            interpret["direction_type_summary"] = {}
            for comp_name, comp_detail in comparisons.items():
                summary = comp_detail.get("summary", {})
                if summary:
                    interpret["direction_type_summary"][comp_name] = {
                        "mean_abs_cosine": summary.get("mean_abs_cosine"),
                        "max_abs_cosine": summary.get("max_abs_cosine"),
                        "max_abs_cosine_layer": summary.get("max_abs_cosine_layer"),
                    }

    return interpret


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_summary(summary: Dict[str, Any], base_name: str) -> None:
    """Print a compact cross-stage summary to console."""
    print()
    print("=" * 80)
    print(f"  SUMMARY: {base_name}")
    print("=" * 80)

    # --- Stage 1: Identify ---
    identify = summary.get("identify", {})
    if identify:
        print(f"\nIDENTIFY:")

        # MC accuracy
        mc = identify.get("mc_accuracy", {})
        if mc and mc.get("overall") is not None:
            n = mc.get("n_questions", "?")
            correct = int(mc["overall"] * n) if isinstance(n, int) else "?"
            print(f"  MC accuracy: {mc['overall']:.1%} ({correct}/{n})")
            per_pos = mc.get("per_position", {})
            if per_pos:
                pos_strs = [f"{k}={v:.0%}" for k, v in sorted(per_pos.items())]
                print(f"    Per-position: {', '.join(pos_strs)}")

        # Per-metric direction quality
        for metric in ALL_METRICS:
            metric_info = identify.get("per_metric", {}).get(metric, {})
            if not metric_info:
                continue

            print(f"\n  {metric}:")
            for method in ["probe", "mean_diff"]:
                m = metric_info.get(method, {})
                if not m:
                    continue
                ci = m.get("best_r2_ci", [None, None])
                r2_str = _fmt_r2(m.get("best_r2"), ci[0], ci[1])
                r_str = f", r={m['best_pearson']:.3f}" if m.get("best_pearson") is not None else ""
                print(f"    {method:12s}: best L{m.get('best_layer', '?')} R²={r2_str}{r_str}")

        # Answer directions
        answer = identify.get("answer", {})
        if answer:
            print(f"\n  Answer directions:")
            for method in ["probe", "mean_diff"]:
                m = answer.get(method, {})
                if not m:
                    continue
                ci = m.get("best_accuracy_ci", [None, None])
                acc_str = _fmt_pct(m.get("best_accuracy"), ci[0], ci[1])
                print(f"    {method:12s}: best L{m.get('best_layer', '?')} accuracy={acc_str} (chance=25%)")

    # --- Stage 2: Meta-task ---
    meta = summary.get("meta", {})
    if meta:
        print(f"\nMETA-TASK ({META_TASK}):")

        # Behavioral
        behavioral = meta.get("behavioral", {})
        if behavioral:
            token_choice = behavioral.get("token_choice", {})
            if token_choice:
                choice_strs = [f"{k}={v}" for k, v in token_choice.items()
                               if k not in ("n_samples", "num_samples")]
                if choice_strs:
                    print(f"  Token choice: {', '.join(choice_strs)}")

            correlations = behavioral.get("correlations", {})
            for metric, corr_data in correlations.items():
                r = corr_data.get("pearson_r")
                ci = corr_data.get("pearson_ci", [None, None])
                p = corr_data.get("pearson_p")
                rho = corr_data.get("spearman_r")
                if r is not None:
                    ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else ""
                    p_str = f", p={p:.2e}" if p is not None else ""
                    rho_str = f", ρ={rho:.3f}" if rho is not None else ""
                    print(f"  Behavioral ({metric}): r={r:.3f}{ci_str}{p_str}{rho_str}")

        # Transfer
        transfer = meta.get("transfer", {})
        for metric in ALL_METRICS:
            t = transfer.get(metric, {})
            if not t:
                continue

            print(f"\n  Transfer ({metric}):")
            d2d = t.get("best_d2d", {})
            if d2d and d2d.get("r2") is not None:
                print(f"    D→D:          best L{d2d.get('layer', '?')} R²={d2d['r2']:.3f}")

            d2m = t.get("best_d2m_centered", {})
            if d2m and d2m.get("r2") is not None:
                ci = d2m.get("r2_ci", [None, None])
                r2_str = _fmt_r2(d2m["r2"], ci[0], ci[1])
                r_str = f", r={d2m['pearson']:.3f}" if d2m.get("pearson") is not None else ""
                print(f"    D→M centered: best L{d2m.get('layer', '?')} R²={r2_str}{r_str}")

            ratio = t.get("transfer_ratio")
            if ratio is not None:
                if ratio > 0.6:
                    strength = "Strong"
                elif ratio > 0.3:
                    strength = "Moderate"
                elif ratio > 0.1:
                    strength = "Weak"
                else:
                    strength = "No"
                print(f"    Transfer ratio: {ratio:.1%} → {strength} evidence")

        # Answer confound
        confound = meta.get("answer_confound", {})
        if confound:
            d2d_acc = confound.get("d2d_accuracy")
            d2m_acc = confound.get("d2m_accuracy")
            if d2d_acc is not None or d2m_acc is not None:
                d2d_str = f"{d2d_acc:.1%}" if d2d_acc is not None else "N/A"
                d2m_str = f"{d2m_acc:.1%}" if d2m_acc is not None else "N/A"
                print(f"\n  Answer confound: D→D={d2d_str}, D→M={d2m_str}")

    # --- Stage 3: Causality ---
    causality = summary.get("causality", {})
    if causality:
        print(f"\nCAUSALITY ({META_TASK}):")

        for stage_type in ["ablation", "steering"]:
            stage_data = causality.get(stage_type, {}).get(META_TASK, {})
            if not stage_data:
                continue

            for metric, metric_data in stage_data.items():
                print(f"\n  {stage_type.title()} ({metric}):")
                for method in ["probe", "mean_diff"]:
                    m = metric_data.get(method, {})
                    if not m:
                        continue
                    n_sig = m.get("n_significant", 0)
                    n_tested = m.get("n_layers_tested", "?")
                    best_layer = m.get("best_layer", "?")
                    z = m.get("best_effect_z")

                    if stage_type == "ablation":
                        effect = m.get("best_effect")
                        effect_str = f"Δcorr={effect:+.4f}" if effect is not None else ""
                        z_str = f", Z={z:.1f}" if z is not None else ""
                        print(f"    {method:12s}: {n_sig}/{n_tested} significant (FDR<0.05), best L{best_layer} {effect_str}{z_str}")
                    else:
                        n_sign = m.get("n_sign_correct", 0)
                        slope = m.get("best_slope")
                        slope_str = f"slope={slope:.4f}" if slope is not None else ""
                        z_str = f", Z={z:.1f}" if z is not None else ""
                        print(f"    {method:12s}: {n_sig}/{n_tested} significant, {n_sign} sign-correct, best L{best_layer} {slope_str}{z_str}")

    # --- Stage 4: Interpret ---
    interpret = summary.get("interpret", {})
    if interpret:
        print(f"\nINTERPRET:")

        # Direction similarity
        dir_sim = interpret.get("direction_similarity", {})
        if dir_sim:
            for pair_key, pair_data in dir_sim.items():
                mean_cos = pair_data.get("mean_abs_cosine")
                max_cos = pair_data.get("max_abs_cosine")
                max_layer = pair_data.get("max_abs_layer")
                if mean_cos is not None:
                    # Shorten pair key for display
                    label = pair_key.replace("__vs__", " · ")
                    print(f"  {label}: mean |cos|={mean_cos:.3f}, max={max_cos:.3f} (L{max_layer})")

        # Direction type comparison
        dir_type = interpret.get("direction_type_summary", {})
        if dir_type:
            print()
            for comp_name, comp_data in dir_type.items():
                mean_cos = comp_data.get("mean_abs_cosine")
                if mean_cos is not None:
                    label = comp_name.replace("_", " ")
                    print(f"  {label}: mean |cos|={mean_cos:.3f}")

        # Interpretations
        interps = interpret.get("direction_type_interpretations", {})
        if interps:
            for key, text in interps.items():
                print(f"  → {key}: {text}")

        # Logit lens (just show primary metric if available)
        logit_lens = interpret.get("logit_lens", {})
        if logit_lens:
            # Show first direction type found
            for dir_name, layers_data in logit_lens.items():
                # Pick one representative layer
                layer_keys = sorted(layers_data.keys(), key=lambda x: int(x))
                if layer_keys:
                    mid_idx = len(layer_keys) // 2
                    mid_layer = layer_keys[mid_idx]
                    tokens = layers_data[mid_layer]
                    token_str = ", ".join(f'"{t}"' for t in tokens[:5])
                    print(f"  Logit lens L{mid_layer} ({dir_name}): {token_str}")

    # Count stages present
    stages_present = sum(1 for k in ["identify", "meta", "causality", "interpret"]
                         if summary.get(k))
    stages_total = 4
    print(f"\n{'─' * 80}")
    print(f"  Stages with data: {stages_present}/{stages_total}")
    if stages_present < stages_total:
        missing = [k for k in ["identify", "meta", "causality", "interpret"]
                   if not summary.get(k)]
        print(f"  Missing: {', '.join(missing)}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_name = _get_base_name()
    model_short = _get_model_short()

    print(f"Summarizing results for: {base_name}")
    print(f"  Model: {MODEL}")
    print(f"  Dataset: {DATASET}")
    print(f"  Meta-task: {META_TASK}")
    print(f"  Primary metric: {PRIMARY_METRIC}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Check what files exist
    all_json = list(OUTPUT_DIR.glob(f"{base_name}*_results.json"))
    # Also check for model-only prefixed files (analyze_directions)
    all_json += list(OUTPUT_DIR.glob(f"{model_short}_direction_*.json"))
    # And causality files which may use different prefix patterns
    all_json += list(OUTPUT_DIR.glob(f"{model_short}*_{META_TASK}_*_results.json"))

    # Deduplicate
    all_json = list(set(all_json))

    print(f"\nFound {len(all_json)} result file(s):")
    for p in sorted(all_json):
        print(f"  {p.name}")

    if not all_json:
        print("\nNo result files found. Run the pipeline scripts first.")
        return

    # Parse each stage
    summary = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            adapter=ADAPTER,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            meta_task=META_TASK,
            primary_metric=PRIMARY_METRIC,
        ),
    }

    identify = parse_identify_results(base_name)
    if identify and (identify.get("per_metric") or identify.get("mc_accuracy")):
        summary["identify"] = identify

    meta = parse_meta_results(base_name, META_TASK)
    if meta:
        summary["meta"] = meta

    causality = parse_causality_results(base_name, META_TASK)
    if causality:
        summary["causality"] = causality

    interpret = parse_interpret_results(base_name)
    if interpret:
        summary["interpret"] = interpret

    # Print console summary
    print_summary(summary, base_name)

    # Save JSON
    summary_path = OUTPUT_DIR / f"{base_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
