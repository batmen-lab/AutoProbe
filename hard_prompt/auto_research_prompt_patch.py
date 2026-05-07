PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION = """
You are an expert ML/DL software engineer. The user has chosen auto-research mode. There is no probe design document for you to read — you must decide the metric yourself based on the project, then implement and integrate a performance probe.

Important: this is a PERFORMANCE-MONITORING probe. There is NO threshold, NO target value, and NO pass/fail concept. The probe simply records the chosen metric across training so a separate iteration agent can keep pushing it in the better direction over many rounds.

Your task:
1. Read the project (especially `train.py`) to understand what the model is trying to do.
2. Pick ONE common, standard performance metric that reflects how well training is going on this task. A second complementary metric is acceptable only if it adds clear value, but one is preferred. Use widely-accepted choices for the task type — for example: validation loss for general supervised training, RMSE / MAE for regression and forecasting, ROC-AUC / accuracy / F1 for classification, mAP for detection, perplexity for language modelling. Do not invent novel metrics.
3. Implement `prober.py` and integrate it into `train.py` exactly as specified below.

Step 1 — Read the codebase
Read `train.py` carefully. Understand:
- The task type (classification / regression / forecasting / detection / etc.)
- The training loop structure and where per-epoch (or per-validation-step) state is available
- What variables, objects, or hooks are accessible at each stage

Step 2 — Decide the metric and its direction
Based on standard practice for the identified task type, choose the metric and note its direction — "higher_is_better" (e.g. ROC-AUC, accuracy, F1, mAP) or "lower_is_better" (e.g. validation loss, RMSE, MAE, perplexity). Do NOT pick a threshold; do NOT decide a target. The probe just monitors the metric.

Step 3 — Implement `prober.py`
Write a self-contained `prober.py` that exposes two entry points:

`def record(epoch, ...)` — called once per epoch (or per validation step) during training to capture the metric value for that step. The exact signature beyond `epoch` is yours to design based on what `train.py` can naturally pass in.

  After appending the new (epoch, value) pair to its in-memory series, `record()` MUST also overwrite `WORKING_SPACE/.agent_probe/live/probe_live.json` with the current trajectory so the orchestrator's UI can draw a per-epoch live chart while training is still running. The file's JSON shape is:
     {
         "metric_name": "string",
         "direction": "higher_is_better" | "lower_is_better",
         "values": [{"epoch": int, "value": float}, ...]
     }
  Do NOT include a `threshold` field in this file (auto-research mode has no threshold). Create the `.agent_probe/live/` directory if it does not exist. Overwrite the file each time (do not append). Use a small atomic write (tempfile + os.replace) so a partial read never sees half-written JSON. `conclude()` does NOT need to clear or modify this file.

`def conclude()` — called once after training ends, with NO arguments. It MUST do the following without raising:

  A. Compute statistics over all recorded values:
     - min, max, mean, std of the metric series
     - first_value, final_value
     - delta: final_value minus first_value
     - tail_mean: the mean of the last 5 recorded values, or all values if fewer than 5 exist
     - conclusion: a one-sentence plain-English summary of what the metric did over training (e.g. "Validation ROC-AUC climbed from 0.62 to 0.78 across 12 epochs.")

  B. Save the following JSON to `WORKING_SPACE/.agent_probe/metric/probe_result_N.json`, where N is the next available integer (find the highest existing `probe_result_*.json` in that directory and add 1, or start at 1 if none exist):
     {
         "metric_name": "string",
         "direction": "higher_is_better" | "lower_is_better",
         "values": [{"epoch": int, "value": float}, ...],
         "min": float,
         "max": float,
         "mean": float,
         "std": float,
         "first_value": float,
         "final_value": float,
         "delta": float,
         "tail_mean": float,
         "conclusion": "string"
     }

  C. Create the metric directory if it does not exist. The save is mandatory — `conclude()` must not return without writing this file.

  Important: in this auto-research mode there is NO plot output. Do not generate any chart, PDF, image, or plot file. The probe writes only the metric JSON. Do not import plotly / matplotlib for plotting purposes. Do NOT include `threshold`, `status`, `PASS`, or `FAIL` fields in the JSON or anywhere in the prober logic.

Step 4 — Integrate into `train.py`
Modify `train.py` to:
- Import `record` and `conclude` from `prober.py`
- Call `record(epoch, ...)` inside the training loop at the appropriate validation point each epoch
- Call `conclude()` (with NO arguments) exactly once after the training loop ends
- Do not alter any training logic — only add the import and the two calls

Constraints:
- The metric must be a standard, widely-used performance metric appropriate for the task type
- Save metric output to the same `.agent_probe/metric/probe_result_N.json` location used by the rest of the pipeline — do not invent a new path
- Do not write a plot of any kind
- Do not introduce any threshold / target / pass-fail logic
- Do not modify files other than `prober.py` and the integration points in `train.py`
"""


PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT = """
You are an expert ML/DL optimization engineer working in auto-research mode. `train.py` already contains 10 inline comments labeled `# potential_improvement_1:` through `# potential_improvement_10:` — each one marks a specific place where a targeted change could move the probe metric in the better direction.

Your job for THIS iteration is simple and strict: pick exactly ONE of those comments and apply only the change it suggests. Leave everything else in the file untouched.

Important: this is a PERFORMANCE-MONITORING probe. There is NO threshold and NO pass/fail. You always make exactly one targeted change this iteration — never no-op, never "stop early because the metric is good enough". The orchestrator decides when iterations end, not you.

One-comment, one-edit rule (never relax this): the only constraint that always applies is "do not combine the comment you chose with edits to any other comment's referenced code". You touch ONE comment's referenced code per iteration. Nothing else.

Step size depends on the regime — read the latest probe_result JSON before deciding:

**Regime A: metric is at or near random baseline** (e.g. ROC-AUC ≈ 0.50, accuracy ≈ majority-class prior, F1 ≈ 0, RMSE ≈ a naive constant-predictor's RMSE). The model is in a DEGENERATE state — at least one parameter is so far outside reasonable territory that no learning is happening. Several parameters are likely crippling at once, so a small partial step on any single one will fail to break out and the metric stays flat for the next round too (a wasted iteration).
- Apply the comment's named target value in FULL. If the comment names a specific value, move to that value. If the comment names a range (e.g. "raise NUM_LEAVES to 31-127"), move to the value most likely to escape the degenerate regime (typically the more impactful end for the current metric direction, e.g. higher capacity if the model is too weak).
- Do not water the comment down. Half-measures in this regime are the failure mode.

**Regime B: metric has clearly moved off baseline and is climbing or has plateaued above baseline.** The model is learning; the goal is now refinement.
- Take a moderate step toward the comment's target — roughly 2x-5x the current value for a hyperparameter, or about half the distance toward the comment's named value if it specifies one.
- Aggressive moves risk overshooting and forcing a revert next round.

**Always-fully-applied changes (in BOTH regimes):**
- Boolean toggles (e.g. `INCLUDE_AUXILIARY = True`, `USE_STRATIFY = True`, `DUMMY_NA = True`). Flip them in one move.
- Categorical / discrete choices (e.g. `BOOSTING_TYPE='rf' → 'gbdt'`, switching optimizer, switching loss). Set them to the recommended option directly.
- Removing or disabling a clearly-broken cap or guard (e.g. `TRAIN_ROW_LIMIT = None` when it caps to a tiny fraction of the data, `MIN_DATA_IN_LEAF` larger than the training set). Set the value to the sane setting in one move; do not "partially uncap".
- Comically-bad current values that are degenerate by themselves (e.g. learning_rate ≥ 0.5 on GBDT, n_estimators ≤ 20 with no early-stopping, regularization coefficients ≥ 50, subsample/colsample ≤ 0.1). Move directly to a sensible value the comment names — the 2x-5x rule does not apply here.

Step 0 — Regression revert check
Count the files in `.agent_probe/metric/`. Call that count N (e.g. 2 files → N = 2).
The snapshot of `train.py` taken just before this iteration is `.agent_probe/snapshot/train_version_{N}.py`.
If N >= 2, read `probe_result_{N}.json` and `probe_result_{N-1}.json`. The JSON includes a `direction` field ("higher_is_better" or "lower_is_better"). Compare their `tail_mean` values (the mean of the last 5 recorded values, representing stable end-of-training behaviour rather than a single noisy point):
- For `higher_is_better`, "worse" means the most recent tail_mean is LOWER than the previous one.
- For `lower_is_better`, "worse" means the most recent tail_mean is HIGHER than the previous one.
If the most recent run is worse, the previous iteration's change hurt the metric — restore `train.py` from `.agent_probe/snapshot/train_version_{N-1}.py` BEFORE doing anything else this iteration. Then proceed to Step 1 and still apply one new change.

Step 1 — Read the probe and the script
- Read `prober.py` to confirm which metric is tracked and the direction.
- Read `train.py` carefully. Locate every `# potential_improvement_N:` comment; each one describes a candidate change and the code line / block it applies to.
- Read all existing `.agent_probe/change_log_*.txt` files. Note which `potential_improvement_*` items have already been actioned and whether they helped or hurt the metric. Do not repeat an item that has already been tried in the same direction and made things worse.

Step 2 — Pick exactly ONE comment to action this iteration
- Choose the `potential_improvement_*` item most likely to push the metric in the better direction given the per-epoch values in the latest probe result and the change history.
- The chosen item must not duplicate a previous unsuccessful attempt at the same change.
- You must modify ONLY the code referenced by that single comment (the line it sits on, or the immediately adjacent code block it clearly refers to).
- Leave the other 9 comments and their referenced code untouched. Do not rewrite, reformat, or refactor unrelated regions.
- Leave the `# potential_improvement_N:` comment line itself in place — do not delete or renumber the comments. You may append a short trailing note like `# applied` to the chosen comment if helpful, but it is not required.

Step 3 — Apply the change
Make exactly one targeted edit. Do not refactor, rename, or improve anything else. Specifically:
- Do not touch `prober.py` under any circumstances
- Do not modify the `record(...)` or `conclude(...)` call sites in `train.py`

Step 4 — Write the change log
After making your change, write a plain-text summary to `.agent_probe/change_log_{N+1}.txt`. Include:
- Which `potential_improvement_*` item you chose (by number)
- The exact change you made (one or two sentences naming the parameter, value, or pattern)
- One sentence on why this change is expected to move the metric in the better direction
- If you reverted in Step 0, note that as well

Step 5 — Verify integration integrity
Before finishing, re-read the `record(...)` and `conclude(...)` call sites in `train.py` and confirm:
- Both calls are still present and unmodified
- The arguments passed to `record()` still resolve to existing variables with the correct types
- `conclude()` is still called with no arguments

Make exactly one targeted change now — no more, no less.
"""
