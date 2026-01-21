#!/usr/bin/env python3
"""Swap-runner v2: run an ablation script under multiple direction variants.

This script is designed to *not* require CLI args. Edit the CONFIG section.

Core safety properties:
- Creates/uses a persistent ORIG_BACKUP of the directions NPZ.
- Always restores the original NPZ contents at the end (even on crash),
  leaving the workspace in a safe state.

Quality-of-life:
- Detects the common failure mode: swapping a different directions NPZ than
  the ablation script actually loads (mismatched INPUT_BASE_NAME / METRIC).
- Produces a compact per-variant summary table (also written to CSV/MD).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np


# =========================
# CONFIG (edit these)
# =========================
WORKDIR = Path("/workspace/entropy_probes")

ABLATION_SCRIPT = WORKDIR / "run_ablation_causality.py"

# This is the BASE NAME used to form the active directions NPZ path:
#   outputs/{INPUT_BASE_NAME}_mc_{METRIC}_directions.npz
# IMPORTANT: this must match what the ablation script is going to use,
# otherwise your variants won't affect anything.
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC"
METRIC = "logit_gap"

# Which vector is being swapped
METHOD = "mean_diff"
TARGET_LAYER = 31

# Variant NPZs produced by your mixture analysis
# (these typically contain ONLY one key for the layer, which we merge into the active NPZ)
VARIANT_PARALLEL_NPZ = (
    WORKDIR / "outputs" / f"{INPUT_BASE_NAME}_mc_{METRIC}_directions_TEST2_parallel_L{TARGET_LAYER}.npz"
)
VARIANT_ORTH_NPZ = (
    WORKDIR / "outputs" / f"{INPUT_BASE_NAME}_mc_{METRIC}_directions_TEST2_orth_L{TARGET_LAYER}.npz"
)

# If True, parse INPUT_BASE_NAME / METRIC from the ablation script and abort on mismatch.
ENFORCE_ABLATION_CONFIG_MATCH = True

# If True, also attempt to parse the ablation script's layer override and warn if it isn't set to TARGET_LAYER.
WARN_IF_LAYER_OVERRIDE_MISMATCH = True


# =========================
# Implementation
# =========================

@dataclass(frozen=True)
class SwapPaths:
    out_dir: Path
    active_npz: Path
    backup_npz: Path
    parallel_npz: Path
    orth_npz: Path


def _now_stamp() -> str:
    # keep sortable timestamps
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _print_banner(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def _hash_vec(vec: np.ndarray) -> str:
    # short stable hash for sanity checks
    h = hashlib.sha256(vec.tobytes()).hexdigest()
    return h[:12]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    # Fix: Ensure temp file ends in .npz so numpy doesn't double-append extension.
    # e.g. path="foo.npz" -> tmp="foo.tmp.npz"
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _expected_key(method: str, layer: int) -> str:
    return f"{method}_layer_{layer}"


def _ensure_backup(paths: SwapPaths) -> None:
    if paths.backup_npz.exists():
        print(f"[swap-runner] Using existing backup: {paths.backup_npz}")
        return
    print(f"[swap-runner] Creating backup: {paths.backup_npz}")
    shutil.copy2(paths.active_npz, paths.backup_npz)


def _restore_original(paths: SwapPaths) -> None:
    shutil.copy2(paths.backup_npz, paths.active_npz)
    print(f"[swap-runner] Restored original directions to: {paths.active_npz}")


def _read_constant_from_py(py_path: Path, name: str) -> Optional[str]:
    """Parse a simple string constant like NAME = 'value' or NAME = \"value\"."""
    text = py_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"^[ \t]*{re.escape(name)}[ \t]*=[ \t]*['\\\"]([^'\\\"]+)['\\\"]",
                  text, flags=re.MULTILINE)
    return m.group(1) if m else None


def _read_layer_override_from_py(py_path: Path) -> Optional[List[int]]:
    """Best-effort parse for a simple list override, e.g. LAYERS = [38] or LAYERS_OVERRIDE = [38]."""
    text = py_path.read_text(encoding="utf-8", errors="replace")
    for name in ("LAYERS", "LAYERS_OVERRIDE", "EXPLICIT_LAYERS", "LAYER_OVERRIDE"):
        m = re.search(rf"^[ \t]*{re.escape(name)}[ \t]*=[ \t]*\[([^\]]*)\]",
                      text, flags=re.MULTILINE)
        if not m:
            continue
        inner = m.group(1).strip()
        if not inner:
            return []
        vals = []
        for tok in inner.split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(int(tok))
                except ValueError:
                    return None
        return vals
    return None


def _check_ablation_script_matches() -> None:
    if not ENFORCE_ABLATION_CONFIG_MATCH and not WARN_IF_LAYER_OVERRIDE_MISMATCH:
        return
    if not ABLATION_SCRIPT.exists():
        raise FileNotFoundError(f"Ablation script not found: {ABLATION_SCRIPT}")

    ab_in = _read_constant_from_py(ABLATION_SCRIPT, "INPUT_BASE_NAME")
    ab_metric = _read_constant_from_py(ABLATION_SCRIPT, "METRIC")

    problems = []
    if ENFORCE_ABLATION_CONFIG_MATCH:
        if ab_in is None:
            problems.append("Could not parse INPUT_BASE_NAME from the ablation script.")
        elif ab_in != INPUT_BASE_NAME:
            problems.append(
                f"INPUT_BASE_NAME mismatch:\n"
                f"  swap-runner INPUT_BASE_NAME = {INPUT_BASE_NAME}\n"
                f"  ablation   INPUT_BASE_NAME = {ab_in}"
            )
        if ab_metric is None:
            problems.append("Could not parse METRIC from the ablation script.")
        elif ab_metric != METRIC:
            problems.append(
                f"METRIC mismatch:\n"
                f"  swap-runner METRIC = {METRIC}\n"
                f"  ablation   METRIC = {ab_metric}"
            )

    if WARN_IF_LAYER_OVERRIDE_MISMATCH:
        layers = _read_layer_override_from_py(ABLATION_SCRIPT)
        if layers is not None and layers != [] and (TARGET_LAYER not in layers):
            print(
                f"[swap-runner][warn] Ablation script layer override looks like {layers}, "
                f"but TARGET_LAYER is {TARGET_LAYER}. You may be running the wrong layer."
            )

    if problems:
        msg = "\n\n".join(problems)
        raise RuntimeError(
            "Config mismatch: you're swapping a different directions NPZ than the ablation script will load.\n" + msg
        )


def _get_paths() -> SwapPaths:
    out_dir = WORKDIR / "outputs"
    active_npz = out_dir / f"{INPUT_BASE_NAME}_mc_{METRIC}_directions.npz"
    backup_npz = out_dir / f"{INPUT_BASE_NAME}_mc_{METRIC}_directions.ORIG_BACKUP.npz"
    return SwapPaths(
        out_dir=out_dir,
        active_npz=active_npz,
        backup_npz=backup_npz,
        parallel_npz=VARIANT_PARALLEL_NPZ,
        orth_npz=VARIANT_ORTH_NPZ,
    )


def _print_active_key(paths: SwapPaths, key: str) -> Tuple[str, float]:
    arrs = _load_npz(paths.active_npz)
    if key not in arrs:
        raise KeyError(f"Active NPZ missing key: {key}")
    v = arrs[key].astype(np.float32, copy=False)
    h = _hash_vec(v)
    n = float(np.linalg.norm(v))
    print(f"[swap-runner] ACTIVE key={key} hash={h} norm={n:.6f}  (expected_npz={paths.active_npz})")
    return h, n


def _swap_in_variant(paths: SwapPaths, variant: str) -> Tuple[str, float]:
    """Mutate the active NPZ for this variant; return (hash, norm) of the target key."""
    key = _expected_key(METHOD, TARGET_LAYER)

    # Always start from clean original file to avoid accumulating swaps.
    _restore_original(paths)

    if variant == "orig":
        # Keep original directions; just report hash/norm.
        return _print_active_key(paths, key)

    if variant == "parallel":
        src = paths.parallel_npz
    elif variant == "orth":
        src = paths.orth_npz
    else:
        raise ValueError(f"Unknown variant: {variant}")

    src_arrs = _load_npz(src)
    if key not in src_arrs:
        raise KeyError(f"Variant NPZ missing key {key}: {src}")

    # Merge: replace only this key, keep all other keys from the active/original NPZ.
    active_arrs = _load_npz(paths.active_npz)
    active_arrs[key] = src_arrs[key]
    _save_npz(paths.active_npz, active_arrs)

    return _print_active_key(paths, key)


def _run_ablation(paths: SwapPaths, run_dir: Path, variant: str) -> Path:
    log_path = run_dir / f"{variant}.log"
    env = os.environ.copy()

    cmd = [sys.executable, str(ABLATION_SCRIPT)]
    print(f"[swap-runner] Running: {' '.join(cmd)}")
    print(f"[swap-runner] CWD: {WORKDIR}")

    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(p.stdout)
        rc = p.returncode

    # Also stream a short tail for convenience.
    tail = "\n".join(p.stdout.splitlines()[-40:])
    print("[swap-runner] --- ablation output tail (last ~40 lines) ---")
    print(tail)
    print("[swap-runner] --- end tail ---")

    if rc != 0:
        raise RuntimeError(f"Ablation script failed (rc={rc}). See log: {log_path}")

    return log_path


def _snapshot_outputs(paths: SwapPaths, run_dir: Path, variant: str) -> Path:
    """Copy the most relevant ablation outputs into a variant-specific artifact folder."""
    art_dir = run_dir / f"{variant}_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "*_ablation_*_results.json",
        "*_ablation_*_checkpoint.json",
        "*_ablation_*_*.png",
    ]
    copied = 0
    for pat in patterns:
        for p in paths.out_dir.glob(pat):
            try:
                shutil.copy2(p, art_dir / p.name)
                copied += 1
            except Exception:
                pass

    print(f"[swap-runner] Snapshotted {copied} files into: {art_dir}")
    return art_dir


_ABLATION_ROW_RE = re.compile(
    # ^ matches start of line; \s* allows indentation
    r"^[ \t]*(?P<layer>\d+)[ \t]+"
    # Base value
    r"(?P<base>[+-]?[0-9]*\.[0-9]+)"
    # Base CI
    r"[ \t]+\[(?P<base_lo>[+-]?[0-9]*\.[0-9]+),[ \t]*(?P<base_hi>[+-]?[0-9]*\.[0-9]+)\]"
    # Ablated value
    r"[ \t]+(?P<abl>[+-]?[0-9]*\.[0-9]+)"
    # Abl CI
    r"[ \t]+\[(?P<abl_lo>[+-]?[0-9]*\.[0-9]+),[ \t]*(?P<abl_hi>[+-]?[0-9]*\.[0-9]+)\]"
    # Delta value
    r"[ \t]+(?P<delta>[+-]?[0-9]*\.[0-9]+)"
    # Delta CI
    r"[ \t]+\[(?P<del_lo>[+-]?[0-9]*\.[0-9]+),[ \t]*(?P<del_hi>[+-]?[0-9]*\.[0-9]+)\]"
    # BootFDR and Hurt flag
    r"[ \t]+(?P<fdr>[0-9]*\.?[0-9]+)[ \t]+(?P<hurt>yes|no)[ \t]*$"
)


def _extract_layer_row_from_log(log_path: Path, layer: int) -> Optional[Dict[str, Any]]:
    """Parse the final summary table row for a given layer from the ablation log."""
    text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(text):
        m = _ABLATION_ROW_RE.match(line)
        if not m:
            continue
        if int(m.group("layer")) != layer:
            continue
        return {
            "layer": int(m.group("layer")),
            "base": float(m.group("base")),
            "base_ci": (float(m.group("base_lo")), float(m.group("base_hi"))),
            "abl": float(m.group("abl")),
            "abl_ci": (float(m.group("abl_lo")), float(m.group("abl_hi"))),
            "delta": float(m.group("delta")),
            "delta_ci": (float(m.group("del_lo")), float(m.group("del_hi"))),
            "bootFDR": float(m.group("fdr")),
            "hurt": m.group("hurt"),
        }
    return None


def _write_compact_summary(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    # CSV
    csv_path = run_dir / "compact_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,layer,base,base_ci_lo,base_ci_hi,abl,abl_ci_lo,abl_ci_hi,delta,delta_ci_lo,delta_ci_hi,bootFDR,hurt\n")
        for r in rows:
            f.write(
                f"{r['variant']},{r['layer']},{r['base']},{r['base_ci'][0]},{r['base_ci'][1]},"
                f"{r['abl']},{r['abl_ci'][0]},{r['abl_ci'][1]},"
                f"{r['delta']},{r['delta_ci'][0]},{r['delta_ci'][1]},"
                f"{r['bootFDR']},{r['hurt']}\n"
            )

    # Markdown
    md_path = run_dir / "compact_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Swap-run summary\n\n")
        f.write(f"Base: `{INPUT_BASE_NAME}`  Metric: `{METRIC}`  Method: `{METHOD}`  Layer: `{TARGET_LAYER}`\n\n")
        f.write("| variant | base corr | abl corr | Δcorr | ΔCI | bootFDR | hurt? |\n")
        f.write("|---|---:|---:|---:|---|---:|:---:|\n")
        for r in rows:
            lo, hi = r["delta_ci"]
            f.write(
                f"| {r['variant']} | {r['base']:+.4f} | {r['abl']:+.4f} | {r['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['bootFDR']:.3f} | {r['hurt']} |\n"
            )

    print(f"[swap-runner] Wrote: {csv_path}")
    print(f"[swap-runner] Wrote: {md_path}")


def main() -> None:
    _check_ablation_script_matches()

    paths = _get_paths()
    if not paths.active_npz.exists():
        raise FileNotFoundError(f"Active directions NPZ not found: {paths.active_npz}")

    key = _expected_key(METHOD, TARGET_LAYER)

    _ensure_backup(paths)

    orig_arrs = _load_npz(paths.backup_npz)
    if key not in orig_arrs:
        raise KeyError(f"Backup NPZ missing key {key}: {paths.backup_npz}")
    
    orig_vec = orig_arrs[key].astype(np.float32, copy=False)
    orig_hash = _hash_vec(orig_vec)
    orig_norm = float(np.linalg.norm(orig_vec))
    print(f"[swap-runner] ORIG  key={key} hash={orig_hash} norm={orig_norm:.6f}")

    run_dir = paths.out_dir / "direction_swap_runs" / f"{INPUT_BASE_NAME}_{METRIC}_{METHOD}_L{TARGET_LAYER}_{_now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = ["orig", "parallel", "orth"]
    summary_rows: List[Dict[str, Any]] = []

    try:
        for variant in variants:
            _print_banner(f"[swap-runner] VARIANT: {variant}")

            h, n = _swap_in_variant(paths, variant)

            log_path = _run_ablation(paths, run_dir, variant)
            art_dir = _snapshot_outputs(paths, run_dir, variant)

            row = _extract_layer_row_from_log(log_path, TARGET_LAYER)
            if row is None:
                print(
                    f"[swap-runner][warn] Could not parse the final summary table row for layer {TARGET_LAYER} "
                    f"from {log_path}. (Table format may have changed.)"
                )
                row = {
                    "layer": TARGET_LAYER,
                    "base": float("nan"),
                    "base_ci": (float("nan"), float("nan")),
                    "abl": float("nan"),
                    "abl_ci": (float("nan"), float("nan")),
                    "delta": float("nan"),
                    "delta_ci": (float("nan"), float("nan")),
                    "bootFDR": float("nan"),
                    "hurt": "?",
                }

            row.update(
                {
                    "variant": variant,
                    "active_key_hash": h,
                    "active_key_norm": n,
                    "log": str(log_path),
                    "artifacts": str(art_dir),
                }
            )
            summary_rows.append(row)

            print(f"[swap-runner] Done variant={variant} key_hash={h} norm={n:.6f}")

    finally:
        _restore_original(paths)

    # Write summary.json
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_base_name": INPUT_BASE_NAME,
                "metric": METRIC,
                "method": METHOD,
                "target_layer": TARGET_LAYER,
                "target_key": key,
                "orig_hash": orig_hash,
                "orig_norm": orig_norm,
                "rows": summary_rows,
            },
            f,
            indent=2,
        )
    print(f"[swap-runner] Summary: {summary_path}")

    # Write compact summaries
    _write_compact_summary(run_dir, summary_rows)

    # Quick sanity check
    deltas = [r.get("delta") for r in summary_rows if isinstance(r.get("delta"), (int, float))]
    if deltas and all(isinstance(d, float) and np.isfinite(d) for d in deltas):
        if max(deltas) - min(deltas) < 1e-6:
            print(
                "\n[swap-runner][WARN] All variants produced identical Δcorr. "
                "If you expected differences, the most common cause is still "
                "a base-name/metric mismatch (you swapped a different NPZ than the ablation script loaded)."
            )

    _print_banner("[swap-runner] RUN COMPLETE")


if __name__ == "__main__":
    main()