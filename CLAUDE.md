# Guidelines for Claude

## Do not run tests locally
This repo runs on a remote machine. Never run `python run_mc_answer_ablation.py` or similar test commands locally - it won't work and wastes time. Rely on static analysis instead.

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

## Critical Codebase Patterns

### Left-Padding
This codebase uses **left-padding** (`core/model_utils.py:160`). When extracting logits from the last token position:
- **CORRECT:** `logits[i, -1, :]` - always gets last position
- **WRONG:** `logits[i, seq_len - 1, :]` - gets wrong position with left-padding

### Options Dict Preservation
Difficulty-filtered datasets store options as a dict `{"A": "...", "B": "...", ...}` with letter keys. The loader in `load_and_format_datasets.py` checks for pre-stored options and uses them directly without reshuffling, preserving exact prompt reproducibility.

### Configuration Pattern
Scripts use module-level constants (MODEL, DATASET, NUM_QUESTIONS, etc.) rather than CLI args. Edit the constants directly to change behavior.
