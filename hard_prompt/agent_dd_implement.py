PROMPT_FIVE = """
PROTECTED CODE - user_analyze() (HARD RULE): If train.py contains a function named `user_analyze` or any block marked "USER ANALYSIS - DO NOT MODIFY", you MUST leave it completely untouched. Do NOT modify, remove, rename, reorder, wrap, gate, comment out, or change its body, its arguments, its call site, or its `.agent_probe/.user_analysis` output. It is a human-owned independent audit and is NOT part of the probe metric. Preserve it verbatim across every edit; make all your changes elsewhere.

ORIGINAL TRAIN METRIC ANCHOR (HARD RULE): The probe metric is NOT the only thing that matters. train.py already has its OWN training objective — a LOSS it minimizes and a primary EVAL / checkpoint-selection metric it reports on validation (e.g. the metric used to pick the best checkpoint). This is the "anchor": the model's original purpose, which the probe must never be allowed to destroy. You MUST:
  (1) Read train.py and identify (a) the training loss and (b) the primary eval / checkpoint-selection metric computed on the validation set. This is train.py's OWN loss/eval — it is NOT `user_analyze()` (that is the human audit; do not treat it as the anchor).
  (2) Mark each at its definition/computation site in train.py with a one-line comment exactly of the form `# ANCHOR: original train metric - do not remove` so later edit-rounds can see it.
  (3) Record the FINAL-EPOCH value of each anchor (measured on the same validation data train.py evaluates on) into the probe_result JSON under the key `original_train_metric` if there is exactly one, or `original_train_metric_0`, `original_train_metric_1`, … if there is more than one. Each entry is an object {"name": string, "value": float, "direction": "higher_is_better" | "lower_is_better"}. Record the primary eval/selection metric first, then the loss. Wire train.py to pass these final-epoch values through to `prober.conclude()` (or have `record()` capture them each epoch and `conclude()` emit the last epoch's). Do NOT fold the anchor into the probe metric — it is a separate, independent record that lives ALONGSIDE the probe fields.
The orchestrator enforces a hard utility floor from these: any later edit that degrades an anchor by more than 20% from the round-1 baseline (lower for a higher-is-better anchor, higher for a lower-is-better anchor) is auto-reverted. Recording them accurately every run is mandatory.

You are an expert ML/DL software engineer. You will receive a development document describing a training-quality probe — including its implementation plan, the metric it computes, the `standard_threshold` (the strict PASS bar), and the `acceptable_threshold` (the loose "we'd settle for this" bar).

Your task is to implement this probe by writing `prober.py` and integrating it into the existing `train.py` in this workspace.

Step 1 — Read the codebase
Read `train.py` to understand:
- The model architecture and training loop structure
- Where training data, validation data, and per-epoch state are available
- What variables, objects, or hooks you can access at each stage of the pipeline

Step 2 — Implement `prober.py`
Write a self-contained `prober.py` that exposes two entry points:

`def record(epoch, ...)` — called once per epoch during training to collect the metric value for that step.

  After appending the new (epoch, value) pair to its in-memory series, `record()` MUST also overwrite `WORKING_SPACE/.agent_probe/live/probe_live.json` with the current trajectory so the orchestrator's UI can draw a per-epoch live chart while training is still running. The file's JSON shape is:
     {
         "metric_name": "string",
         "standard_threshold": float,
         "acceptable_threshold": float,
         "direction": "higher_is_better" | "lower_is_better",
         "values": [{"epoch": int, "value": float}, ...]
     }
  Create the `.agent_probe/live/` directory if it does not exist. Overwrite the file each time (do not append). Use a small atomic write (tempfile + os.replace) so a partial read never sees half-written JSON. `conclude()` does NOT need to clear or modify this file — leaving the final trajectory in place is correct.

`def conclude(standard_threshold, acceptable_threshold)` — called once after training completes with BOTH threshold values. It must do the following with no exceptions:

  A. Compute statistics over all recorded values:
     - min, max, mean, std of the metric series
     - delta: final_value minus first_value (positive = improving toward higher-is-better threshold, negative = degrading)
     - tail_mean: the mean of the last 5 recorded values (or all values if fewer than 5 exist)
     - status: "PASS" if tail_mean satisfies the `standard_threshold` condition (≥ for higher_is_better, ≤ for lower_is_better), "FAIL" otherwise. The `acceptable_threshold` does NOT affect status — only the standard threshold does. The acceptable threshold is recorded for the orchestrator's downstream "best we can realistically do" check.
     - acceptable_met: boolean — true iff tail_mean satisfies the `acceptable_threshold` condition (same direction logic as PASS). Computed and stored separately from status.
     - conclusion: a one-sentence plain-English summary of what the probe found (e.g. "Validation F1 improved steadily from 0.42 to 0.71, crossing the 0.65 standard threshold at epoch 14.")

  B. Save the following JSON to `WORKING_SPACE/.agent_probe/metric/probe_result_N.json`, where N is the next available integer (1, 2, 3, …) — i.e. find the highest existing probe_result_*.json in that directory and increment by 1, starting at 1 if none exist:
     {
         "metric_name": "string",
         "standard_threshold": float,
         "acceptable_threshold": float,
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
         "status": "PASS" | "FAIL",
         "acceptable_met": bool,
         "conclusion": "string",
         "original_train_metric": {"name": "string", "value": float, "direction": "higher_is_better" | "lower_is_better"}
         // ^ the anchor (train.py's own final-epoch loss/eval). If train.py has MORE THAN ONE
         //   primary loss/eval metric, OMIT "original_train_metric" and instead include one
         //   object per anchor: "original_train_metric_0": {...}, "original_train_metric_1": {...}.
         //   Record the primary eval/selection metric first, the loss next. This is a separate,
         //   independent record — do NOT let it change how the probe metric itself is computed.
     }

  C. Generate a Plotly line chart and save it as `WORKING_SPACE/.agent_probe/plot/probe_result_N.pdf`, using the same N as the JSON file above.
     The chart must include:
     - A labeled line for the metric values over epochs
     - A red horizontal dashed line for the `standard_threshold`, annotated with its value (labelled "standard")
     - An amber/orange horizontal dashed line for the `acceptable_threshold`, annotated with its value (labelled "acceptable")
     - A vertical annotation (or marker) at the epoch where the metric first crosses the `standard_threshold` (if it does)
     - Chart title: the metric name
     - X-axis label: "Epoch"
     - Y-axis label: the metric name
     - A text box in the chart showing: min, max, mean, std, delta, trend, status, and acceptable_met
     - Color the metric line green if status is PASS, red if FAIL
     - Requirement: fix an appropriate range for all axes after the first chart is generated, so that all subsequent charts have the same y-axis range for comparability — the range must include both threshold values and leave headroom for future iterations.
  D. Create both output directories if they do not exist. These saves are mandatory — conclude() must not return without writing both files.

Step 3 — Integrate into `train.py`
Modify `train.py` to:
- Import `record` and `conclude` from `prober.py`
- Call `record(epoch, ...)` inside the training loop at the appropriate point each epoch
- Call `conclude(standard_threshold, acceptable_threshold)` once after the training loop ends, passing BOTH threshold values from the development document (in that order)
- Do not alter the training logic — only add the import and the two calls

Development document:
"""
