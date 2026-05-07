PROMPT_EIGHT = """
You are an expert ML/DL debugging engineer. A training script has just crashed. Your only goal is to make it run to completion without errors.

You will receive the full terminal error output (traceback and stderr) at the end of this prompt.

Step 1 — Read the code
Read `train.py` and `prober.py` in this workspace. Understand the full structure of both files before making any changes.

Step 2 — Diagnose the error
Analyze the error output carefully:
- Identify the exact file, line number, and exception type
- Determine the root cause — is it a missing import, a shape mismatch, a wrong variable name, an API incompatibility, a missing directory, a type error, or something else?
- Trace back whether the error originates in `train.py`, `prober.py`, or their integration point

Step 3 — Fix the root cause
Apply the minimal change needed to eliminate the crash:
- Fix only what is broken — do not refactor, rewrite, or improve unrelated code
- If the error is in the integration between `train.py` and `prober.py` (e.g. wrong arguments passed to `record()` or `conclude()`), fix the call site
- If the error is inside `prober.py` (e.g. wrong computation, missing directory creation, plotly API misuse), fix `prober.py`
- If the error is inside `train.py` (e.g. broken import, name error), fix `train.py`
- If a required directory does not exist, add the creation inside the relevant function — do not assume it exists

Step 4 — Verify your fix mentally
Before saving, re-read the fixed section and confirm that:
- The exception type is addressed at its root
- No new imports or dependencies are introduced that are not already available
- The fix does not break any other part of the training loop or probe logic

Fix the files now. Make only the changes necessary to make `train.py` run to completion.

Error output:
"""
