"""
Logging utilities for entropy probes experiments.

Provides standardized console output (minimal) and log file output (detailed).
All scripts should use these utilities for consistent output formatting.

Console Output Pattern:
    ================================================================================
    SCRIPT_NAME: Stage N - Description
    ================================================================================
    Config: model={model}, dataset={dataset}, task={task}

    [tqdm progress bar]

    Key Findings:
      Best R²: 0.567 [0.521, 0.613] at layer 24
      Transfer: 0.412, Significant layers: 12/32

    Output: {base}_mc_results.json, {base}_mc_results.png
    Log: {base}_mc_run.log
    ================================================================================

Usage:
    from core.logging_utils import setup_run_logger, print_run_header, print_run_footer
    from core.model_utils import get_model_dir_name

    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    logger = setup_run_logger(None, base_name, "mc", model_dir=model_dir)
    print_run_header("identify_mc_correlate.py", 1, "Find MC uncertainty directions", config)

    # ... do work, use logger.info() for detailed output ...
    logger.info(f"Layer {layer}: R²={r2:.3f}")

    print_run_footer(output_files, logger.log_file)
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import subprocess

from core.config_utils import get_output_path


def _get_git_hash() -> str:
    """Get short git hash of current commit, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


class RunLogger:
    """Logger that writes to both console (minimal) and file (detailed)."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self._logger = logging.getLogger(f"entropy_probes.{log_file.stem}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        # File handler - detailed output
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self._logger.addHandler(fh)

        # No console handler - we use print for console output
        self._logger.propagate = False

    def info(self, msg: str):
        """Log info-level message (goes to file only)."""
        self._logger.info(msg)

    def debug(self, msg: str):
        """Log debug-level message (goes to file only)."""
        self._logger.debug(msg)

    def warning(self, msg: str):
        """Log warning (goes to file only)."""
        self._logger.warning(msg)

    def section(self, title: str):
        """Log a section header."""
        self._logger.info(f"\n{'=' * 80}")
        self._logger.info(f"[SECTION: {title}]")
        self._logger.info('=' * 80)

    def table(self, headers: List[str], rows: List[List[Any]], title: Optional[str] = None):
        """Log a formatted table."""
        if title:
            self._logger.info(f"\n{title}:")

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Format header
        header_line = "  ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
        self._logger.info(f"  {header_line}")
        self._logger.info(f"  {'-' * len(header_line)}")

        # Format rows
        for row in rows:
            row_line = "  ".join(f"{str(cell):<{widths[i]}}" for i, cell in enumerate(row))
            self._logger.info(f"  {row_line}")

    def dict(self, data: Dict[str, Any], title: Optional[str] = None, indent: int = 2):
        """Log a dictionary as key-value pairs."""
        if title:
            self._logger.info(f"\n{title}:")
        prefix = " " * indent
        for key, value in data.items():
            if isinstance(value, float):
                self._logger.info(f"{prefix}{key}: {value:.6f}")
            else:
                self._logger.info(f"{prefix}{key}: {value}")

    def per_layer_results(self, results: Dict[int, Dict[str, Any]], metrics: List[str]):
        """Log per-layer results table."""
        self.section("Per-Layer Results")

        for metric in metrics:
            if metric not in ["r2", "accuracy"]:
                continue

            headers = ["Layer", metric.upper()]
            if "ci_low" in str(results.get(0, {})):
                headers.extend(["CI Low", "CI High"])

            rows = []
            for layer in sorted(results.keys()):
                layer_data = results[layer]
                row = [layer, f"{layer_data.get(metric, 0):.4f}"]
                if f"{metric}_ci_low" in layer_data:
                    row.extend([
                        f"{layer_data[f'{metric}_ci_low']:.4f}",
                        f"{layer_data[f'{metric}_ci_high']:.4f}"
                    ])
                rows.append(row)

            self.table(headers, rows, f"{metric.upper()} by Layer")


def setup_run_logger(
    output_dir: Path,
    base_name: str,
    script_name: str,
    model_dir: str = None,
) -> RunLogger:
    """
    Create a logger for a run.

    Args:
        output_dir: Directory for output files (ignored - uses centralized path management)
        base_name: Base name for output files (e.g., "model_dir_dataset" or just "dataset")
        script_name: Short script identifier (e.g., "mc", "meta_confidence")
        model_dir: Optional model directory for output path routing

    Returns:
        RunLogger that writes to outputs/{model_dir}/working/{base_name}_{script_name}_run.log
    """
    # Use centralized path management - log files go to working/
    log_file = get_output_path(f"{base_name}_{script_name}_run.log", model_dir=model_dir)
    logger = RunLogger(log_file)

    # Write header to log file
    logger.info("=" * 80)
    logger.info(f"ENTROPY PROBES LOG: {script_name}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Git Hash: {_get_git_hash()}")
    logger.info("=" * 80)

    return logger


def print_run_header(
    script_name: str,
    stage: int,
    description: str,
    config: Dict[str, Any],
):
    """
    Print minimal run header to console.

    Args:
        script_name: Name of the script (e.g., "identify_mc_correlate.py")
        stage: Stage number (1-4)
        description: One-line description
        config: Configuration dict with model, dataset, etc.
    """
    print("=" * 80)
    print(f"{script_name}: Stage {stage} - {description}")
    print("=" * 80)

    # Print condensed config
    config_items = []
    for key in ["model", "dataset", "task", "metric", "direction_type"]:
        if key in config:
            value = config[key]
            # Shorten model names
            if key == "model" and "/" in str(value):
                value = str(value).split("/")[-1]
            config_items.append(f"{key}={value}")

    if config_items:
        print(f"Config: {', '.join(config_items)}")
    print()


def print_key_finding(label: str, value: str, indent: int = 2):
    """Print a single key finding."""
    print(f"{' ' * indent}{label}: {value}")


def print_key_findings(findings: Dict[str, str]):
    """
    Print key findings section.

    Args:
        findings: Dict mapping finding labels to values
                  e.g., {"Best R²": "0.567 [0.521, 0.613] at layer 24"}
    """
    print("Key Findings:")
    for label, value in findings.items():
        print_key_finding(label, value)
    print()


def print_run_footer(output_files: List[Union[str, Path]], log_file: Optional[Path] = None):
    """
    Print minimal run footer to console.

    Args:
        output_files: List of output file paths
        log_file: Path to the log file (optional)
    """
    # Condense to just filenames
    filenames = [Path(f).name for f in output_files]

    if len(filenames) <= 3:
        print(f"Output: {', '.join(filenames)}")
    else:
        print(f"Output: {filenames[0]}, {filenames[1]}, ... ({len(filenames)} files)")

    if log_file:
        print(f"Log: {Path(log_file).name}")

    print("=" * 80)


def format_r2_with_ci(r2: float, ci_low: Optional[float] = None, ci_high: Optional[float] = None) -> str:
    """Format R² value with optional confidence interval."""
    if ci_low is not None and ci_high is not None:
        return f"{r2:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
    return f"{r2:.3f}"


def format_best_layer(method: str, layer: int, r2: float, ci_low: Optional[float] = None, ci_high: Optional[float] = None) -> str:
    """Format best layer result for console output."""
    r2_str = format_r2_with_ci(r2, ci_low, ci_high)
    return f"{method}: R²={r2_str} at layer {layer}"
