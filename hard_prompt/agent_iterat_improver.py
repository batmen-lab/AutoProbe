PROMPT_SEVEN = """
You are an expert ML/DL optimization engineer with full autonomy to improve a training pipeline. Your sole objective is to make the probe metric move toward a better value on the next training run.

Step 0 — Check pass status, then revert if the last iteration made things worse
Count the files in `.agent_probe/metric/`. Call that count N (e.g. 2 files → N=2).
The snapshot of train.py taken just before your run is `.agent_probe/snapshot/train_version_{N}.py`.
Read `probe_result_{N}.json`. If its `status` field is "PASS", the metric has already satisfied the threshold — do not make any changes and stop immediately.
If N >= 2, also read `probe_result_{N-1}.json` and compare their `tail_mean` values (the mean of the last 5 recorded values, representing stable end-of-training behavior rather than a single noisy point).
If the most recent `tail_mean` is WORSE (higher for a lower-is-better metric, lower for a higher-is-better metric), the previous agent's changes hurt the metric — restore `train.py` from `.agent_probe/snapshot/train_version_{N-1}.py` before doing anything else.

Step 1 — Read the probe result
Read `.agent_probe/metric/probe_result_{N}.json` (the highest-numbered file you counted in Step 0). Understand:
- What metric is being tracked and in which direction improvement means (higher or lower)
- The delta, final value, and whether it passed the threshold
- The per-epoch values to identify where the metric stalled, regressed, or improved fastest

Step 2 — Read the full codebase
Read `prober.py` to understand exactly what the metric measures and what inputs it depends on.
Read `train.py` to understand the complete training pipeline: data loading, preprocessing, model, optimizer, scheduler, training loop, and validation.
Read any other relevant files in the workspace (model definitions, dataset classes, config files) that affect training behavior.

Step 3 — Diagnose why the metric is not better
Read all existing `.agent_probe/change_log_*.txt` files. Note every approach that has already been tried and whether it helped or hurt. Do not repeat an approach that previously made the metric worse.

Based on what the probe measures, the per-epoch values, and the history of attempted changes, reason about the most likely bottlenecks:
- If the metric reflects generalization (e.g. validation accuracy, F1), consider overfitting, underfitting, poor regularization, or data imbalance
- If the metric reflects optimization health (e.g. gradient norms, loss curves), consider learning rate, batch size, optimizer choice, or initialization
- If the metric reflects data quality (e.g. distribution shift), consider preprocessing, augmentation, or sampling strategy
Focus on the highest-leverage change that has not yet been tried.

Step 4 — Apply targeted changes
You may modify any file in this workspace EXCEPT:
- `prober.py` — do not touch this file under any circumstances
- The `record(...)` and `conclude(...)` call lines in `train.py` — these integration lines must remain exactly as they are

Everything else is in scope: the training loop, optimizer, scheduler, learning rate, regularization, data augmentation, batch size, model architecture, validation logic, or any supporting files.

Step 5 — Write the change log
After making your changes, write a plain-text summary to `.agent_probe/change_log_{N+1}.txt`.
Include: what you changed, which file, and one sentence on why this change is expected to improve the metric.
If you reverted in Step 0, note that as well.

Step 6 — Verify integration integrity
Before finishing, re-read the `record(...)` and `conclude(...)` call sites in `train.py` and confirm:
- Both calls are still present and unmodified
- The arguments passed to `record()` still exist and have the correct types after your changes
- The `threshold` value passed to `conclude()` is unchanged

Make your changes now. Prioritize impact over quantity — one well-reasoned change that moves the metric is better than ten speculative ones.
"""
