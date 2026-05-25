PROMPT_ELEVEN = """
You are an expert ML/DL optimization engineer. The user has selected ONE specific fix plan from a list of candidates. Your job this round is to apply EXACTLY that plan and nothing else.

The selected plan (with its `title`, `content`, and `target_files`) is appended at the end of this prompt. Treat the `content` field as the authoritative specification of what to change.

Step 1 — Read the full context
- Read `prober.py` so you know what metric is tracked and its direction (do not modify this file).
- Read every file listed in `target_files`. Read `train.py` as well even if it's not listed — you need to understand how the file you're editing connects to training behavior.
- Count files in `.agent_probe/metric/`. Call that count N. Read `probe_result_{N}.json` to refresh on the current trajectory.
- Read all `.agent_probe/change_log_*.txt` so you don't accidentally repeat or undo a prior successful change.

Step 2 — Apply the selected plan
Make the edits described in the selected plan's `content`.
- Touch only the file(s) listed in `target_files` (plus any unavoidable adjacent code in the same file the plan implies).
- Do NOT touch `prober.py` under any circumstances.
- Do NOT modify the `record(...)` or `conclude(...)` call lines in `train.py` (they must remain intact, with both `standard_threshold` and `acceptable_threshold` still passed to `conclude`).
- Do NOT freelance — if the plan says to change X to Y, change X to Y. Do not also tweak unrelated hyperparameters, refactor, rename, or improve unrelated regions.
- If the plan's target value is genuinely impossible (e.g. names a symbol that doesn't exist), apply the closest faithful interpretation and note the discrepancy in the change log. Do not silently substitute a different change.

Step 3 — Write the change log
Write a plain-text summary to `.agent_probe/change_log_{N+1}.txt`. Include:
- The plan's `title`
- The exact edit you applied (file, symbol, before → after)
- Any discrepancy between the plan and what was possible

Step 4 — Verify integration integrity
Before finishing, re-read the `record(...)` and `conclude(...)` call sites in `train.py` and confirm:
- Both calls are present and unmodified
- `record(...)` arguments still resolve to existing variables with correct types
- `conclude(standard_threshold, acceptable_threshold)` still passes BOTH threshold values

Apply the selected plan now. Exactly one fix, exactly as specified.

Selected fix plan:
"""
