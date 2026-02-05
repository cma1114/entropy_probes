"""
Standard visualization helpers for entropy probes experiments.

Provides consistent colors, figure sizes, and plotting functions used across
all experiment scripts. Import helpers from here to ensure visual consistency.

Usage:
    from core.plotting import (
        METHOD_COLORS, DIRECTION_COLORS, TASK_COLORS,
        plot_layer_metric, mark_significant_layers, save_figure,
        plot_metric_distributions, plot_directions_summary,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# =============================================================================
# STANDARD COLORS
# =============================================================================

# Direction-finding method colors (used in R² plots, transfer plots, comparisons)
METHOD_COLORS = {
    "probe": "tab:blue",
    "mean_diff": "tab:orange",
    "centroid": "tab:orange",  # answer direction equivalent of mean_diff
}

# Direction type colors (used in compare_direction_types, ablation type comparisons)
DIRECTION_COLORS = {
    "uncertainty": "tab:blue",
    "answer": "tab:green",
    "confidence": "tab:red",
}

# Meta-task colors (used in multi-task comparison plots)
TASK_COLORS = {
    "confidence": "tab:purple",
    "delegate": "tab:cyan",
    "other_confidence": "tab:gray",
}

# Significance coloring for FDR-corrected p-values
SIGNIFICANCE_COLORS = {
    "sig_005": "red",       # p < 0.05
    "sig_010": "orange",    # p < 0.10
    "ns": "gray",           # not significant
}

# Condition colors for ablation/steering
CONDITION_COLORS = {
    "baseline": "blue",
    "ablated": "red",
    "control": "gray",
    "steered": "green",
}

# Metric colors for multi-metric plots
METRIC_COLORS = {
    "entropy": "tab:purple",
    "top_prob": "tab:green",
    "margin": "tab:red",
    "logit_gap": "tab:blue",
    "top_logit": "tab:cyan",
}

# =============================================================================
# STANDARD FIGURE SIZES
# =============================================================================

SINGLE_PANEL = (10, 5)
TWO_PANEL_WIDE = (14, 5)
THREE_PANEL_WIDE = (18, 5)
THREE_PANEL_TALL = (20, 14)
TWO_PANEL_TALL = (20, 10)
FOUR_PANEL = (16, 10)

# =============================================================================
# DISPLAY CONSTANTS
# =============================================================================

R2_FLOOR = -0.5   # Clip R² below this for display
R2_CEIL = 1.0     # Max R²
DPI = 300          # Standard output DPI
GRID_ALPHA = 0.3   # Standard grid transparency
CI_ALPHA = 0.2     # Standard confidence interval band transparency
MARKER_SIZE = 4    # Standard marker size for line plots
LINE_WIDTH = 1.5   # Standard line width


# =============================================================================
# PLOTTING HELPERS
# =============================================================================


def get_method_color(method: str) -> str:
    """Get the standard color for a direction-finding method."""
    return METHOD_COLORS.get(method, "tab:gray")


def get_significance_color(p_value: float, alpha: float = 0.05, marginal_alpha: float = 0.10) -> str:
    """Get color based on significance level."""
    if p_value < alpha:
        return SIGNIFICANCE_COLORS["sig_005"]
    elif p_value < marginal_alpha:
        return SIGNIFICANCE_COLORS["sig_010"]
    return SIGNIFICANCE_COLORS["ns"]


def significance_colors_for_layers(p_values: List[float], alpha: float = 0.05) -> List[str]:
    """Map a list of p-values to significance colors."""
    return [get_significance_color(p, alpha) for p in p_values]


def plot_layer_metric(
    ax: plt.Axes,
    values: Union[List[float], np.ndarray],
    method: str = "probe",
    label: Optional[str] = None,
    ci_low: Optional[np.ndarray] = None,
    ci_high: Optional[np.ndarray] = None,
    layers: Optional[List[int]] = None,
    marker: str = "o",
):
    """
    Plot a metric across layers with consistent styling.

    Args:
        ax: Matplotlib axes to plot on
        values: Metric values per layer
        method: Direction-finding method name (determines color)
        label: Legend label (defaults to method name)
        ci_low: Lower confidence interval bound per layer
        ci_high: Upper confidence interval bound per layer
        layers: Layer indices for x-axis (defaults to 0..n-1)
        marker: Marker style
    """
    color = get_method_color(method)
    x = layers if layers is not None else list(range(len(values)))
    label = label or method

    ax.plot(x, values, f"{marker}-", label=label, color=color,
            markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

    if ci_low is not None and ci_high is not None:
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=CI_ALPHA)


def mark_significant_layers(
    ax: plt.Axes,
    layers: List[int],
    values: Union[List[float], np.ndarray],
    p_values: List[float],
    alpha: float = 0.05,
    color: Optional[str] = None,
    marker: str = "*",
    size: int = 80,
):
    """
    Mark significant layers with star markers on an existing plot.

    Args:
        ax: Matplotlib axes
        layers: Layer indices
        values: Y-values at each layer
        p_values: P-values for each layer
        alpha: Significance threshold
        color: Marker color (default: red)
        marker: Marker style
        size: Marker size
    """
    color = color or SIGNIFICANCE_COLORS["sig_005"]
    sig_x = [l for l, p in zip(layers, p_values) if p < alpha]
    sig_y = [v for v, p in zip(values, p_values) if p < alpha]
    if sig_x:
        ax.scatter(sig_x, sig_y, color=color, s=size, marker=marker, zorder=5,
                   edgecolor="black", linewidth=0.5)


def add_best_layer_annotation(
    ax: plt.Axes,
    best_layer: int,
    best_value: float,
    metric_name: str = "R²",
    offset: Tuple[float, float] = (10, 10),
):
    """
    Add an annotation arrow pointing to the best layer.

    Args:
        ax: Matplotlib axes
        best_layer: Layer index of the best value
        best_value: The metric value at the best layer
        metric_name: Name of the metric for the annotation text
        offset: Text offset in points from the point
    """
    ax.annotate(
        f"L{best_layer}: {metric_name}={best_value:.3f}",
        xy=(best_layer, best_value),
        xytext=offset,
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def add_significance_legend(ax: plt.Axes, alpha: float = 0.05, loc: str = "best"):
    """Add a standard significance legend to a bar chart."""
    legend_elements = [
        Patch(facecolor=SIGNIFICANCE_COLORS["sig_005"], alpha=0.7, edgecolor="black",
              label=f"FDR < {alpha}"),
        Patch(facecolor=SIGNIFICANCE_COLORS["sig_010"], alpha=0.7, edgecolor="black",
              label=f"FDR < {alpha * 2}"),
        Patch(facecolor=SIGNIFICANCE_COLORS["ns"], alpha=0.7, edgecolor="black",
              label="n.s."),
    ]
    ax.legend(handles=legend_elements, loc=loc, fontsize=9)


def format_layer_axis(
    ax: plt.Axes,
    layers: List[int],
    xlabel: str = "Layer",
    ylabel: str = "",
    title: str = "",
    grid: bool = True,
):
    """Apply standard formatting to a layer-indexed axis."""
    x = np.arange(len(layers))
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=GRID_ALPHA)


def add_summary_text_panel(
    ax: plt.Axes,
    text: str,
    fontsize: int = 10,
    bg_color: str = "white",
):
    """Turn an axes into a text-only summary panel."""
    ax.axis("off")
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor=bg_color, edgecolor="gray", alpha=0.9),
    )


def clip_r2_for_display(values: np.ndarray, floor: float = R2_FLOOR) -> np.ndarray:
    """Clip R² values for display purposes (very negative R² is meaningless)."""
    return np.clip(values, floor, R2_CEIL)


# =============================================================================
# FIGURE MANAGEMENT
# =============================================================================


def save_figure(fig: plt.Figure, output_path: Union[str, Path], dpi: int = DPI):
    """Save figure with standard settings and close it."""
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_figure(
    n_panels: int = 1,
    orientation: str = "wide",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "",
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create a figure with standard sizing.

    Args:
        n_panels: Number of subplots
        orientation: "wide" (horizontal) or "tall" (vertical)
        figsize: Override figure size
        title: Suptitle for the figure

    Returns:
        (fig, axes) tuple
    """
    if figsize is None:
        if n_panels == 1:
            figsize = SINGLE_PANEL
        elif n_panels == 2:
            figsize = TWO_PANEL_WIDE if orientation == "wide" else TWO_PANEL_TALL
        elif n_panels == 3:
            figsize = THREE_PANEL_WIDE if orientation == "wide" else THREE_PANEL_TALL
        else:
            figsize = FOUR_PANEL

    if orientation == "wide":
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes[0]  # Return 1D array
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
        axes = axes[:, 0]  # Return 1D array

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    if n_panels == 1:
        return fig, axes[0]
    return fig, axes


# =============================================================================
# MC CONSOLIDATED PLOTTING FUNCTIONS
# =============================================================================


def get_metric_label(metric: str, metric_info: Optional[Dict] = None) -> str:
    """Get a human-readable axis label for a metric based on METRIC_INFO."""
    if metric_info is None:
        return metric
    higher_means = metric_info.get("higher_means", "")
    if higher_means == "more_confident":
        return f"{metric} (→ more confident)"
    elif higher_means == "more_uncertain":
        return f"{metric} (→ more uncertain)"
    return metric


def plot_metric_distributions(
    metrics_dict: Dict[str, np.ndarray],
    metadata: List[Dict],
    metric_info_map: Dict[str, Dict],
    output_path: Path,
    title_prefix: str = "MC",
):
    """
    Plot metric distributions with correctness breakdown.

    Creates a figure with one row per metric, each row having 3 panels:
    1. Overall histogram with mean/median markers
    2. By-correctness breakdown (correct vs incorrect)
    3. Accuracy vs metric bins (calibration view)

    Args:
        metrics_dict: {metric_name: values_array} for each metric to plot
        metadata: List of dicts with "is_correct" and metric values per sample
        metric_info_map: {metric_name: {"higher_means": ..., "linear": ...}}
        output_path: Where to save the figure
        title_prefix: Prefix for the figure suptitle
    """
    metrics = list(metrics_dict.keys())
    n_metrics = len(metrics)

    if n_metrics == 0:
        return

    fig, axes = plt.subplots(n_metrics, 3, figsize=(15, 4 * n_metrics), squeeze=False)

    for row, metric in enumerate(metrics):
        values = metrics_dict[metric]
        info = metric_info_map.get(metric, {})
        higher_means = info.get("higher_means", "more_confident")

        # Panel 1: Overall distribution
        ax1 = axes[row, 0]
        ax1.hist(values, bins=30, edgecolor='black', alpha=0.7,
                 weights=np.ones(len(values)) / len(values) * 100,
                 color=METRIC_COLORS.get(metric, "tab:gray"))
        ax1.axvline(values.mean(), color='red', linestyle='--',
                    label=f'Mean: {values.mean():.3f}')
        ax1.axvline(np.median(values), color='orange', linestyle='--',
                    label=f'Median: {np.median(values):.3f}')
        ax1.set_xlabel(get_metric_label(metric, info))
        ax1.set_ylabel('Percentage')
        ax1.set_title(f'{metric} Distribution (n={len(values)})')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=GRID_ALPHA)

        # Panel 2: By correctness
        ax2 = axes[row, 1]
        correct_values = [m[metric] for m in metadata if m["is_correct"]]
        incorrect_values = [m[metric] for m in metadata if not m["is_correct"]]

        if correct_values:
            ax2.hist(correct_values, bins=20, alpha=0.6,
                     label=f'Correct (n={len(correct_values)})',
                     color='green',
                     weights=np.ones(len(correct_values)) / len(correct_values) * 100)
        if incorrect_values:
            ax2.hist(incorrect_values, bins=20, alpha=0.6,
                     label=f'Incorrect (n={len(incorrect_values)})',
                     color='red',
                     weights=np.ones(len(incorrect_values)) / len(incorrect_values) * 100)
        ax2.set_xlabel(get_metric_label(metric, info))
        ax2.set_ylabel('Percentage')
        ax2.set_title(f'{metric} by Correctness')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=GRID_ALPHA)

        # Panel 3: Accuracy vs metric bins
        ax3 = axes[row, 2]
        n_bins = 10
        bins = np.linspace(values.min(), values.max(), n_bins + 1)
        bin_accuracies = []
        bin_centers = []
        bin_counts = []

        for i in range(n_bins):
            if i == n_bins - 1:
                bin_mask = (values >= bins[i]) & (values <= bins[i + 1])
            else:
                bin_mask = (values >= bins[i]) & (values < bins[i + 1])

            bin_items = [m for j, m in enumerate(metadata) if bin_mask[j]]
            if len(bin_items) > 0:
                acc = sum(1 for m in bin_items if m["is_correct"]) / len(bin_items)
                bin_accuracies.append(acc)
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_counts.append(len(bin_items))

        ax3.bar(bin_centers, bin_accuracies, width=(bins[1] - bins[0]) * 0.8,
                alpha=0.7, edgecolor='black', color=METRIC_COLORS.get(metric, "tab:gray"))
        ax3.set_xlabel(get_metric_label(metric, info))
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs ' + metric)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=GRID_ALPHA)

        # Add count labels
        for x, y, c in zip(bin_centers, bin_accuracies, bin_counts):
            ax3.text(x, y + 0.02, f'n={c}', ha='center', va='bottom', fontsize=7)

        # Add expected trend annotation
        if higher_means == "more_confident":
            trend_text = "Expected: ↗ (higher = more confident)"
        else:
            trend_text = "Expected: ↘ (higher = more uncertain)"
        ax3.text(0.98, 0.02, trend_text, transform=ax3.transAxes,
                 ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)

    fig.suptitle(f'{title_prefix} Metric Distributions', fontsize=14, fontweight='bold')
    save_figure(fig, output_path)


def plot_directions_summary(
    uncertainty_results: Dict[str, Dict],  # {metric: {"fits": ..., "directions": ..., "comparison": ...}}
    answer_results: Optional[Dict],  # {"fits": ..., "directions": ..., "comparison": ...}
    metrics_dict: Dict[str, np.ndarray],
    metadata: List[Dict],
    metric_info_map: Dict[str, Dict],
    output_path: Path,
    title_prefix: str = "MC",
):
    """
    Create a 6-panel summary figure showing all direction-finding results.

    Panel 1: Probe methods (R²/accuracy vs layer)
    Panel 2: Mean-diff/Centroid methods (R²/accuracy vs layer)
    Panel 3: Direction cosine similarities (probe vs mean_diff, answer probe vs centroid)
    Panel 4: Uncertainty vs Answer direction similarities (probe↔ans_probe, mean_diff↔ans_centroid)
    Panel 5: Pairwise similarities between metric probe directions
    Panel 6: Pairwise similarities between metric mean_diff directions

    Args:
        uncertainty_results: Per-metric direction finding results with "directions" key
        answer_results: Answer direction finding results with "directions" key (optional)
        metrics_dict: {metric: values} (unused, kept for API compatibility)
        metadata: Sample metadata (unused, kept for API compatibility)
        metric_info_map: METRIC_INFO-style dict (unused, kept for API compatibility)
        output_path: Where to save the figure
        title_prefix: Prefix for suptitle
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    metrics = list(uncertainty_results.keys())

    if not metrics:
        return

    # Get layers from first metric
    first_metric = metrics[0]
    layers = sorted(uncertainty_results[first_metric]["fits"]["probe"].keys())

    # =========================================================================
    # Panel 1: Probe methods
    # =========================================================================
    ax1 = axes[0, 0]

    # Plot uncertainty probe R² for each metric
    for metric in metrics:
        fits = uncertainty_results[metric]["fits"]["probe"]
        r2_values = [fits[l]["r2"] for l in layers]
        color = METRIC_COLORS.get(metric, "tab:gray")

        # CIs if available (using r2_std like the old plotting code)
        if "r2_std" in fits[layers[0]]:
            r2_std = [fits[l]["r2_std"] for l in layers]
            ax1.fill_between(layers, np.array(r2_values) - np.array(r2_std),
                           np.array(r2_values) + np.array(r2_std), alpha=CI_ALPHA, color=color)

        best_layer = max(layers, key=lambda l: fits[l]["r2"])
        best_r2 = fits[best_layer]["r2"]
        ax1.plot(layers, r2_values, 'o-', color=color, markersize=MARKER_SIZE,
                 label=f'{metric} probe (best L{best_layer}: {best_r2:.3f})')

    # Plot answer probe accuracy if available
    if answer_results and "probe" in answer_results["fits"]:
        fits = answer_results["fits"]["probe"]
        acc_values = [fits[l]["test_accuracy"] for l in layers]

        if "test_accuracy_std" in fits[layers[0]]:
            acc_std = [fits[l]["test_accuracy_std"] for l in layers]
            ax1.fill_between(layers, np.array(acc_values) - np.array(acc_std),
                           np.array(acc_values) + np.array(acc_std),
                           alpha=CI_ALPHA, color="black")

        best_layer = max(layers, key=lambda l: fits[l]["test_accuracy"])
        best_acc = fits[best_layer]["test_accuracy"]
        ax1.plot(layers, acc_values, 's--', color="black", markersize=MARKER_SIZE,
                 label=f'answer probe (best L{best_layer}: {best_acc:.1%})')

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    if answer_results:
        ax1.axhline(y=0.25, color='red', linestyle=':', alpha=0.3, label='chance (25%)')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('R² / Accuracy')
    ax1.set_title('Probe Methods')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=GRID_ALPHA)

    # =========================================================================
    # Panel 2: Mean-diff / Centroid methods
    # =========================================================================
    ax2 = axes[0, 1]

    # Plot uncertainty mean_diff R² for each metric
    for metric in metrics:
        fits = uncertainty_results[metric]["fits"]["mean_diff"]
        r2_values = [fits[l]["r2"] for l in layers]
        color = METRIC_COLORS.get(metric, "tab:gray")

        if "r2_std" in fits[layers[0]]:
            r2_std = [fits[l]["r2_std"] for l in layers]
            ax2.fill_between(layers, np.array(r2_values) - np.array(r2_std),
                           np.array(r2_values) + np.array(r2_std), alpha=CI_ALPHA, color=color)

        best_layer = max(layers, key=lambda l: fits[l]["r2"])
        best_r2 = fits[best_layer]["r2"]
        ax2.plot(layers, r2_values, 'o-', color=color, markersize=MARKER_SIZE,
                 label=f'{metric} mean_diff (best L{best_layer}: {best_r2:.3f})')

    # Plot answer centroid accuracy if available
    if answer_results and "centroid" in answer_results["fits"]:
        fits = answer_results["fits"]["centroid"]
        acc_values = [fits[l]["test_accuracy"] for l in layers]

        if "test_accuracy_std" in fits[layers[0]]:
            acc_std = [fits[l]["test_accuracy_std"] for l in layers]
            ax2.fill_between(layers, np.array(acc_values) - np.array(acc_std),
                           np.array(acc_values) + np.array(acc_std),
                           alpha=CI_ALPHA, color="black")

        best_layer = max(layers, key=lambda l: fits[l]["test_accuracy"])
        best_acc = fits[best_layer]["test_accuracy"]
        ax2.plot(layers, acc_values, 's--', color="black", markersize=MARKER_SIZE,
                 label=f'answer centroid (best L{best_layer}: {best_acc:.1%})')

    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    if answer_results:
        ax2.axhline(y=0.25, color='red', linestyle=':', alpha=0.3, label='chance (25%)')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('R² / Accuracy')
    ax2.set_title('Mean-Diff / Centroid Methods')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=GRID_ALPHA)

    # =========================================================================
    # Panel 3: Direction similarities (probe vs mean_diff)
    # =========================================================================
    ax3 = axes[0, 2]

    # Answer probe vs centroid similarity
    if answer_results and "comparison" in answer_results:
        cos_sims = [answer_results["comparison"][l]["cosine_sim"] for l in layers]
        ax3.plot(layers, cos_sims, 'o-', color="black", markersize=MARKER_SIZE,
                 label=f'answer probe↔centroid (mean: {np.mean(cos_sims):.3f})')

    # Uncertainty probe vs mean_diff similarity per metric
    for metric in metrics:
        if "comparison" in uncertainty_results[metric] and uncertainty_results[metric]["comparison"]:
            cos_sims = [uncertainty_results[metric]["comparison"][l]["cosine_sim"] for l in layers]
            color = METRIC_COLORS.get(metric, "tab:gray")
            ax3.plot(layers, cos_sims, 's-', color=color, markersize=MARKER_SIZE,
                     label=f'{metric} probe↔mean_diff (mean: {np.mean(cos_sims):.3f})')

    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Direction Similarities')
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=GRID_ALPHA)

    # =========================================================================
    # Panel 4: Uncertainty vs Answer direction similarities
    # =========================================================================
    ax4 = axes[1, 0]

    if answer_results and "directions" in answer_results:
        # Plot similarity between each uncertainty direction and corresponding answer direction
        # probe → answer probe, mean_diff → answer centroid
        for metric in metrics:
            color = METRIC_COLORS.get(metric, "tab:gray")

            # Probe vs answer probe
            if "probe" in uncertainty_results[metric]["directions"] and "probe" in answer_results["directions"]:
                probe_sims = []
                for layer in layers:
                    unc_dir = uncertainty_results[metric]["directions"]["probe"][layer]
                    ans_dir = answer_results["directions"]["probe"][layer]
                    cos_sim = float(np.dot(unc_dir, ans_dir))
                    probe_sims.append(cos_sim)
                ax4.plot(layers, probe_sims, 'o-', color=color, markersize=MARKER_SIZE,
                         label=f'{metric} probe↔ans_probe (mean: {np.mean(probe_sims):.3f})')

            # Mean_diff vs answer centroid
            if "mean_diff" in uncertainty_results[metric]["directions"] and "centroid" in answer_results["directions"]:
                md_sims = []
                for layer in layers:
                    unc_dir = uncertainty_results[metric]["directions"]["mean_diff"][layer]
                    ans_dir = answer_results["directions"]["centroid"][layer]
                    cos_sim = float(np.dot(unc_dir, ans_dir))
                    md_sims.append(cos_sim)
                ax4.plot(layers, md_sims, 's--', color=color, markersize=MARKER_SIZE,
                         label=f'{metric} mean_diff↔ans_centroid (mean: {np.mean(md_sims):.3f})')

        ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Cosine Similarity')
        ax4.set_title('Uncertainty vs Answer Direction Similarities')
        ax4.set_ylim(-1.1, 1.1)
        ax4.legend(fontsize=8, loc='best')
        ax4.grid(True, alpha=GRID_ALPHA)
    else:
        # No answer directions available - show placeholder
        ax4.text(0.5, 0.5, 'No answer directions available',
                 transform=ax4.transAxes, ha='center', va='center',
                 fontsize=12, style='italic', alpha=0.5)
        ax4.set_title('Uncertainty vs Answer Direction Similarities')
        ax4.grid(True, alpha=GRID_ALPHA)

    # =========================================================================
    # Panel 5: Pairwise similarities between metric probe directions
    # =========================================================================
    ax5 = axes[1, 1]

    if len(metrics) >= 2:
        # Generate distinct colors for each pair
        pair_colors = plt.cm.tab10.colors
        pair_idx = 0
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if j <= i:
                    continue  # Skip self-comparisons and duplicates
                if "probe" in uncertainty_results[m1]["directions"] and "probe" in uncertainty_results[m2]["directions"]:
                    sims = []
                    for layer in layers:
                        dir1 = uncertainty_results[m1]["directions"]["probe"][layer]
                        dir2 = uncertainty_results[m2]["directions"]["probe"][layer]
                        cos_sim = float(np.dot(dir1, dir2))
                        sims.append(cos_sim)
                    color = pair_colors[pair_idx % len(pair_colors)]
                    ax5.plot(layers, sims, 'o-', color=color, markersize=MARKER_SIZE,
                             label=f'{m1}↔{m2} (mean: {np.mean(sims):.3f})')
                    pair_idx += 1

        ax5.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax5.set_xlabel('Layer')
        ax5.set_ylabel('Cosine Similarity')
        ax5.set_title('Pairwise Probe Direction Similarities')
        ax5.set_ylim(-1.1, 1.1)
        ax5.legend(fontsize=8, loc='best')
        ax5.grid(True, alpha=GRID_ALPHA)
    else:
        ax5.text(0.5, 0.5, 'Need ≥2 metrics for pairwise comparison',
                 transform=ax5.transAxes, ha='center', va='center',
                 fontsize=12, style='italic', alpha=0.5)
        ax5.set_title('Pairwise Probe Direction Similarities')
        ax5.grid(True, alpha=GRID_ALPHA)

    # =========================================================================
    # Panel 6: Pairwise similarities between metric mean_diff directions
    # =========================================================================
    ax6 = axes[1, 2]

    if len(metrics) >= 2:
        pair_idx = 0
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if j <= i:
                    continue
                if "mean_diff" in uncertainty_results[m1]["directions"] and "mean_diff" in uncertainty_results[m2]["directions"]:
                    sims = []
                    for layer in layers:
                        dir1 = uncertainty_results[m1]["directions"]["mean_diff"][layer]
                        dir2 = uncertainty_results[m2]["directions"]["mean_diff"][layer]
                        cos_sim = float(np.dot(dir1, dir2))
                        sims.append(cos_sim)
                    color = pair_colors[pair_idx % len(pair_colors)]
                    ax6.plot(layers, sims, 's-', color=color, markersize=MARKER_SIZE,
                             label=f'{m1}↔{m2} (mean: {np.mean(sims):.3f})')
                    pair_idx += 1

        ax6.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax6.set_xlabel('Layer')
        ax6.set_ylabel('Cosine Similarity')
        ax6.set_title('Pairwise Mean-Diff Direction Similarities')
        ax6.set_ylim(-1.1, 1.1)
        ax6.legend(fontsize=8, loc='best')
        ax6.grid(True, alpha=GRID_ALPHA)
    else:
        ax6.text(0.5, 0.5, 'Need ≥2 metrics for pairwise comparison',
                 transform=ax6.transAxes, ha='center', va='center',
                 fontsize=12, style='italic', alpha=0.5)
        ax6.set_title('Pairwise Mean-Diff Direction Similarities')
        ax6.grid(True, alpha=GRID_ALPHA)

    fig.suptitle(f'{title_prefix} Directions Summary', fontsize=14, fontweight='bold')
    save_figure(fig, output_path)
