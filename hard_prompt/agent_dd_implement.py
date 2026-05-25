PROMPT_FIVE = """
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
         "conclusion": "string"
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
