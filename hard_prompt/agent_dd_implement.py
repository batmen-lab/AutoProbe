PROMPT_FIVE = """
You are an expert ML/DL software engineer. You will receive a development document describing a training-quality probe — including its implementation plan, the metric it computes, and the threshold that separates healthy from problematic.

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
         "threshold": float,
         "direction": "higher_is_better" | "lower_is_better",
         "values": [{"epoch": int, "value": float}, ...]
     }
  Create the `.agent_probe/live/` directory if it does not exist. Overwrite the file each time (do not append). Use a small atomic write (tempfile + os.replace) so a partial read never sees half-written JSON. `conclude()` does NOT need to clear or modify this file — leaving the final trajectory in place is correct.

`def conclude(threshold)` — called once after training completes. It must do the following with no exceptions:

  A. Compute statistics over all recorded values:
     - min, max, mean, std of the metric series
     - delta: final_value minus first_value (positive = improving toward higher-is-better threshold, negative = degrading)
     - tail_mean: the mean of the last 5 recorded values (or all values if fewer than 5 exist)
     - status: "PASS" if tail_mean satisfies the threshold condition, "FAIL" otherwise
     - conclusion: a one-sentence plain-English summary of what the probe found (e.g. "Validation F1 improved steadily from 0.42 to 0.71, crossing the 0.65 threshold at epoch 14.")

  B. Save the following JSON to `WORKING_SPACE/.agent_probe/metric/probe_result_N.json`, where N is the next available integer (1, 2, 3, …) — i.e. find the highest existing probe_result_*.json in that directory and increment by 1, starting at 1 if none exist:
     {
         "metric_name": "string",
         "threshold": float,
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
         "conclusion": "string"
     }

  C. Generate a Plotly line chart and save it as `WORKING_SPACE/.agent_probe/plot/probe_result_N.pdf`, using the same N as the JSON file above.
     The chart must include:
     - A labeled line for the metric values over epochs
     - A horizontal dashed line for the threshold, annotated with its value
     - A vertical annotation (or marker) at the epoch where the metric first crosses the threshold (if it does)
     - Chart title: the metric name
     - X-axis label: "Epoch"
     - Y-axis label: the metric name
     - A text box in the chart showing: min, max, mean, std, delta, trend, and status
     - Color the metric line green if status is PASS, red if FAIL
     - Requirement: fix an appropriate range(what is approprate range is you have to include potential value in later plotting instead of letting data or points out of plot)for all axies of after first chart is generated, so that all subsequent charts have the same y-axis range for comparability
     - Requirement: add a red horizontal line indicate the threshold value, and annotate it with the threshold value
  D. Create both output directories if they do not exist. These saves are mandatory — conclude() must not return without writing both files.

Step 3 — Integrate into `train.py`
Modify `train.py` to:
- Import `record` and `conclude` from `prober.py`
- Call `record(epoch, ...)` inside the training loop at the appropriate point each epoch
- Call `conclude(threshold)` once after the training loop ends, passing the threshold value from the development document
- Do not alter the training logic — only add the import and the two calls

Development document:
"""
