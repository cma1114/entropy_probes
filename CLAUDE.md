# Guidelines for Claude

## Do not run tests locally
This repo runs on a remote machine. Never run `python run_ablation_causality.py` or similar test commands locally - it won't work and wastes time. Rely on static analysis instead.

## Local outputs directory is not representative
The `outputs/` directory on the local machine may be empty or stale. The user runs experiments on a remote GPU machine, so:
- Don't assume files exist or don't exist based on local `ls` or `find`
- Scripts should always check for cached files (activations, directions, etc.) and load them if present
- Never claim "no cached activations exist" based on local inspection

## Complete the task on the first attempt
When asked to do something, implement the full solution - don't wait for follow-up requests to add obvious components. If measuring an effect, include statistical significance. If adding a mode, make it replace the old behavior rather than running both.

## Follow existing patterns
Before writing new code, read how similar things are done in the codebase. Match the existing style for configuration, output format, and code structure. Don't introduce new patterns when established ones exist.

## Don't add unnecessary complexity
Prefer simple solutions. Don't add command-line arguments when module-level constants work. Don't add abstraction layers that aren't needed. Don't run redundant analyses.

## Anticipate what's needed to interpret results
Think about what someone would need to actually use your output. Raw numbers without context are rarely useful. If the user will obviously need X to make sense of Y, include X without being asked.

## When unsure, ask - don't guess conservatively
If the scope is unclear, ask for clarification rather than implementing a minimal version and waiting for complaints. One good implementation is better than three rounds of fixes.

## Solve problems, don't explain them away
When something isn't working, focus on finding and fixing the root cause. Don't suggest workarounds, don't explain why the problem "isn't really a bug", and don't recommend avoiding the broken feature. Keep investigating until you understand what's actually wrong and fix it.

## Self-review before declaring any script complete

After writing or revising a script, STOP and critically review it before telling the user it's ready:

1. **Check for train/test leakage**: If computing a "baseline" or "upper bound", verify train and test sets are separate. Training on data you'll evaluate on is a bug.

2. **Trace what each variable actually contains**: Don't assume - verify. If `within_r` is supposed to be an upper bound, confirm the code actually computes it that way.

3. **Question suspiciously good results**: If cross-dataset transfer reaches 100% of within-dataset, or any metric seems too perfect, assume there's a bug until proven otherwise.

4. **Check the comparison is fair**: When comparing method A vs method B, ensure both are evaluated on the same held-out data with equivalent methodology.

5. **Read the code as a skeptic**: Pretend you're reviewing someone else's code and looking for flaws. What would you criticize?

## Statistical/ML Code Review Checklist

Before declaring any statistical or ML code complete, explicitly verify:

1. **Complexity**: What's the time complexity? If bootstrap, am I refitting models or just recomputing metrics from fixed predictions? Refitting is almost always wrong.

2. **Object identity**: Trace every scaler/pca/model - where was it created? If I call `scaler_shuf, pca_shuf, probe = train_probe(X, y_shuffled)`, I must use `scaler_shuf` and `pca_shuf` later, not objects from a different training run.

3. **Formulas**: Write out symbolically and verify. Common errors:
   - OLS slope = `corr × (σ_y/σ_x)`, NOT `(σ_y/σ_x) × sign(corr)`
   - If computing a metric two ways (e.g., `test_r2` and bootstrapped `test_r2_mean`), they must measure the same thing

4. **Numerical stability**: What happens when variance → 0? For bounded metrics (R² ∈ (-∞,1], accuracy ∈ [0,1]), is output actually bounded? Bootstrap resampling can create pathological samples.

5. **Consistency check**: If I compute `metric` and `metric_mean` (bootstrapped), they should be approximately equal. If they differ systematically, there's a bug.

## Before declaring any refactoring complete

After ANY change, you must SHOW these outputs (not just promise to do them):

1. **Grep for old names**: Run and show the grep output. If it finds anything, you're not done.

2. **Trace data consumers**: List every function that consumes the data you changed. Show grep output for each key/variable name.

3. **Check mode branches**: This codebase has multiple modes:
   - `probe_type`: "mc_classification" vs "entropy_regression"
   - `full_matrix_mode`: True vs False
   - `REVERSE_MODE`: True vs False

   Show grep for `if.*classification|if.*regression|if.*full_matrix|if.*REVERSE` and confirm each branch handles your change.

4. **Code cannot be tested locally** - this repo runs remotely. So you must be extra thorough with static analysis. Trace through the code paths manually.

5. **One crash = full audit**: If the user reports a crash after you said it was ready, grep for ALL similar patterns and fix them all. Show the grep output.

## Read Code, Don't Assume

Before explaining WHY code works a certain way, SHOW the code. If you catch yourself saying "X implies Y" or "the context makes Z redundant", STOP and verify with grep/read.

**Warning signs you're about to make a lazy error:**
- Explaining why something exists without citing line numbers
- Justifications that sound logical but aren't grounded in actual code
- Moving quickly through multiple files without reading deeply
- Reframing the user's requirement to match an easier implementation

**Before changing any naming convention or pattern:**
1. Grep for ALL occurrences of the thing being changed
2. Read the code that USES it, not just the code that DEFINES it
3. State the actual reason it exists (with line numbers), not an assumed reason
4. If you can't find a reason, ASK rather than assuming it's unnecessary

**Before changing any output file format:**
1. Grep for the filename pattern across all scripts
2. Read how each consumer parses/uses the file
3. List all consumers that need updating
4. Update ALL consumers before declaring complete

**If the user expresses frustration:**
1. STOP making changes immediately
2. Read code more carefully, not less
3. Ask clarifying questions rather than guessing
4. Show reasoning with specific line numbers before implementing

## Cross-Script File Dependencies

When adding or modifying features that involve one script producing output that another script consumes:

1. **Trace the producer**: Find the exact line where the output filename is constructed. Note the pattern.
2. **Trace all consumers**: Grep for scripts that read this file. Find the exact line where they construct the expected filename.
3. **Verify exact match**: The patterns must match character-for-character, including any prefixes/suffixes like `_transfer_`, `_meta_`, etc.

**When user reports "file not found" or similar:**
- NEVER assume the file doesn't exist or needs to be generated first
- ALWAYS trace the actual filename being constructed in both producer and consumer
- The bug is almost always a filename mismatch, not a missing file

## Critical Codebase Patterns

### Left-Padding
This codebase uses **left-padding** (`core/model_utils.py:160`). When extracting logits from the last token position:
- **CORRECT:** `logits[i, -1, :]` - always gets last position
- **WRONG:** `logits[i, seq_len - 1, :]` - gets wrong position with left-padding

### Options Dict Preservation
Difficulty-filtered datasets store options as a dict `{"A": "...", "B": "...", ...}` with letter keys. The loader in `load_and_format_datasets.py` checks for pre-stored options and uses them directly without reshuffling, preserving exact prompt reproducibility.

### Configuration Pattern
Scripts use module-level constants (MODEL, DATASET, NUM_QUESTIONS, etc.) rather than CLI args. Edit the constants directly to change behavior. Constants that must be consistent across scripts are marked `# Must match across scripts` (see Shared Parameters below).

### Shared Parameter Contract
These constants must be identical across all scripts that use them:

| Parameter | Default | Scripts |
|-----------|---------|---------|
| `SEED` | `42` | All |
| `PROBE_ALPHA` | `1000.0` | identify_mc, identify_nexttoken, test_meta, test_cross_dataset |
| `PROBE_PCA_COMPONENTS` | `100` | Same as above |
| `TRAIN_SPLIT` | `0.8` | Same + ablation, steering |
| `MEAN_DIFF_QUANTILE` | `0.25` | identify_mc, identify_nexttoken, test_meta, test_cross_dataset |

If you change one, grep for that constant name across all root `.py` files and change them all.

### Output File Naming
Output filenames use `{dataset}` as the base (model prefix is now in the directory). Stage 2 files use `meta_` prefix:
- `{dataset}_meta_{task}_transfer_results.json` - D→M transfer results
- `{dataset}_meta_{task}_transfer_{pos}.png` - Transfer plots (consolidated, all metrics)
- `{dataset}_meta_{task}_activations.npz` - Cached meta activations
- `{dataset}_meta_{task}_confdir_*.{npz,json,png}` - Confidence direction results
- `{dataset}_meta_{task}_mcuncert_*.{npz,json,png}` - MC uncertainty direction results (consolidated, all metrics)

### Output Directory Structure
Outputs are organized by model, then by **purpose** (not file type):
```
outputs/
├── Llama-3.1-8B-Instruct/              # Model directory (from get_model_dir_name)
│   ├── results/                         # Human-readable final outputs (summary JSONs, plots)
│   │   └── TriviaMC_mc_results.json
│   └── working/                         # Machine data (activations, directions, checkpoints, logs)
│       ├── TriviaMC_mc_activations.npz
│       └── *_checkpoint.json            # Checkpoints go here (use working=True)
├── Llama-3.1-8B-Instruct_4bit/         # Quantized model
│   ├── results/
│   └── working/
├── results/                             # Legacy (no model_dir)
└── working/                             # Legacy (no model_dir)
```

**Key distinction:** `results/` is for outputs intended for humans to read/view. `working/` is for machine data (checkpoints, cached computations, intermediate files). Don't route by file extension alone—checkpoints are `.json` but go to `working/`.

Use centralized path helpers from `core/config_utils.py`:
- `get_model_dir_name(model, adapter, load_in_4bit, load_in_8bit)` → model directory name
- `get_output_path(filename, model_dir=, working=False)` → routes to correct subdirectory for writes. Use `working=True` for checkpoints and machine data.
- `find_output_file(filename, model_dir=)` → checks new location first, falls back to legacy for reads
- `glob_outputs(pattern, model_dir=)` → searches across output locations
- `discover_model_dirs()` → lists all model directories

**Standard pattern for primary pipeline scripts:**
```python
from core import get_model_dir_name
from core.config_utils import get_output_path, find_output_file

# At start of main():
model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
base_name = DATASET  # No model prefix - it's in the directory

# For writes:
get_output_path(f"{base_name}_mc_results.json", model_dir=model_dir)
# → outputs/Llama-3.1-8B-Instruct/results/TriviaMC_mc_results.json

# For reads (with migration fallback):
find_output_file(f"{base_name}_mc_results.json", model_dir=model_dir)
```

Do NOT use `OUTPUT_DIR = Path("outputs")` in new code. The helpers handle directory creation automatically.

### Quantization in Model Naming
`get_model_short_name()` and `get_model_dir_name()` in `core/model_utils.py` append `_4bit` or `_8bit` when quantization is enabled. Adapter info is also included: `Llama-3.1-8B-Instruct_4bit_adapter-lora`. When adding new scripts, use `get_model_dir_name()` to construct the model directory name - this ensures quantization and adapter info are reflected in the output path.

### Confidence Interval Format
All CIs use `[lo, hi]` format (percentile-based bootstrap), not `±std`. Example: `R²=0.567 [0.521, 0.613]`. When adding new console output, follow this convention.

### Centralized Visualization (`core/plotting.py`)
All plotting uses helpers from `core/plotting.py`:
- **Colors**: `METHOD_COLORS` (probe=tab:blue, mean_diff=tab:orange, centroid=tab:orange), `DIRECTION_COLORS`, `TASK_COLORS`, `SIGNIFICANCE_COLORS`, `CONDITION_COLORS`
- **Constants**: `DPI=300`, `GRID_ALPHA=0.3`, `CI_ALPHA=0.2`, `MARKER_SIZE=4`, `LINE_WIDTH=1.5`
- **Helpers**: `save_figure(fig, path)` handles tight_layout + savefig + close + print. `plot_layer_metric()`, `mark_significant_layers()`, `format_layer_axis()`, etc.

When writing new plotting code:
- Import colors/constants from `core.plotting` instead of hardcoding
- Use `save_figure(fig, path)` instead of `plt.tight_layout(); plt.savefig(); plt.close(); print()`
- Use `GRID_ALPHA` for `ax.grid(True, alpha=...)` and `CI_ALPHA` for `ax.fill_between(..., alpha=...)`
- Exception: `test_cross_dataset_transfer.py` defines its own `CI_ALPHA = 0.05` for Fisher-z statistical CIs — do not shadow it with the plotting constant

### JSON Config Metadata
Every `*_results.json` includes a `config` section produced by `get_config_dict()` from `core/config_utils.py`. This helper adds timestamp, git hash, and quantization automatically. Pass script-specific constants as keyword arguments.

### Script Consolidation
- `identify_mc_answer_correlate.py` is merged into `identify_mc_correlate.py` (controlled by `FIND_ANSWER_DIRECTIONS` flag)
- `identify_confidence_correlate.py` is merged into `test_meta_transfer.py` (controlled by `FIND_CONFIDENCE_DIRECTIONS` flag)
- `summarize_results.py`, `compare_direction_types.py`, `regenerate_directions_plot.py` moved to `archive/` (redundant)
- The standalone scripts remain in `archive/` for reference
- Pre-modification snapshots are in `archive/originals/`

### Output Consolidation (format_version: 2)
Scripts produce consolidated JSON outputs with `format_version: 2`:
- **Stage 1 MC**: `*_mc_results.json` with sections `{config, dataset, metrics.{metric}, answer}`
- **Stage 1 Nexttoken**: `*_nexttoken_results.json` with sections `{config, dataset, metrics.{metric}, token}`
- Per-layer details go to `*_run.log` files (not console)

Console output follows minimal pattern:
```
================================================================================
SCRIPT: Stage N - Description
================================================================================
Config: model=X, dataset=Y

[tqdm progress bar]

Key Findings:
  Best R²: 0.567 [0.521, 0.613] at layer 24

Output: base_mc_results.json, ... (N files)
Log: base_mc_run.log
================================================================================
```

### Logging Utilities (`core/logging_utils.py`)
All scripts use centralized logging:
- `setup_run_logger(output_dir, base_name, script_name, model_dir=None)` → creates `*_run.log`
- `print_run_header(script_name, stage, description, config)` → console header
- `print_key_findings(findings_dict)` → console summary
- `print_run_footer(output_files, log_file)` → console footer
- Logger methods: `logger.section()`, `logger.info()`, `logger.dict()`, `logger.table()`

### Module Docstring Format
Every active script's docstring includes: purpose, inputs (file patterns loaded), outputs (file patterns produced), shared parameters note, and dependency chain (which scripts must run first).
