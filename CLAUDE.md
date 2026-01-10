# Guidelines for Claude

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
