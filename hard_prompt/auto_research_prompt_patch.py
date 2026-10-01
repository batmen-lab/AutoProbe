PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION = """
PROTECTED CODE - user_analyze() (HARD RULE): If train.py contains a function named `user_analyze` or any block marked "USER ANALYSIS - DO NOT MODIFY", you MUST leave it completely untouched. Do NOT modify, remove, rename, reorder, wrap, gate, comment out, or change its body, its arguments, its call site, or its `.agent_probe/.user_analysis` output. It is a human-owned independent audit and is NOT part of the probe metric. Preserve it verbatim; make all your changes elsewhere.

You are an expert ML/DL software engineer. The user has chosen auto-research mode. There is no probe design document for you to read — you determine the metric from the project itself, then implement and integrate a performance probe. "From the project itself" is nearly always literal: the metric is one `train.py` already computes (see the METRIC RULE below), not one you choose on taste.

Important: this is a PERFORMANCE-MONITORING probe. There is NO threshold, NO target value, and NO pass/fail concept. The probe simply records the tracked metric across training so the stage-4 improvement loop — which drafts candidate fix plans, scores them and applies one per round — can keep pushing it in the better direction over many rounds.

Your task:
1. Read the project (especially `train.py`) to understand what the model is trying to do.
2. Decide which metric to track, following the METRIC RULE below in strict tier order.

   METRIC RULE — TIER 1: REUSE THE METRIC `train.py` IS ALREADY IMPROVING (overwhelmingly preferred; this is the
   default and you need a concrete, stated reason to leave it).
   `train.py` almost always already computes the metric it is being optimised and selected on: the value it keys
   early stopping / "best checkpoint" / model selection on, the headline eval number it prints or logs each epoch or
   each validation step, or the number it writes into a results/metrics file at the end. THAT number is the metric
   this probe tracks. Do not invent a new metric, do not substitute a "better" or more fashionable one, and do not
   redefine, re-aggregate or re-threshold the existing one. Find it in this order:
     a. the value used for model selection / early stopping / best-checkpoint tracking;
     b. the headline eval value printed or logged per epoch / per validation step;
     c. the value written into a results / metrics / summary file at the end of training.
   If several qualify, take the one `train.py` itself treats as primary (a selection metric beats a merely-printed
   extra). Reuse its exact definition, averaging/aggregation and direction as `train.py` computes it, and pass the
   already-computed number straight through to `record()` — do NOT recompute it your own way, even if you believe
   your way is more correct. Reusing the script's own number is the point: the whole run is about pushing the metric
   this project already cares about, and a reimplementation silently changes what is being optimised.

   METRIC RULE — TIER 2: AUPRC (only when Tier 1 genuinely does not apply — `train.py` computes no eval/selection
   metric at all, e.g. it only prints a training loss with no validation evaluation anywhere). Then, if AUPRC (area
   under the precision-recall curve, a.k.a. average precision) CAN be used for this task, you MUST use AUPRC. AUPRC
   applies whenever the task is classification with a well-defined positive class and the model produces per-example
   scores/probabilities — binary classification, and via one-vs-rest / per-class averaging, multi-label and
   multi-class classification. In those cases choose AUPRC over ROC-AUC, accuracy, or F1 (it is far more informative
   under class imbalance, which is common).

   METRIC RULE — TIER 3: only if neither tier applies, pick ONE common, standard performance metric appropriate to
   the task type — validation loss for general supervised training, RMSE / MAE for regression and forecasting, mAP
   for detection, perplexity for language modelling. Do not invent novel metrics.

   In every tier: one metric is preferred; a second complementary metric is acceptable only if it adds clear value.
   State in `prober.py`'s module docstring which tier you used and, for Tier 1, exactly which line of `train.py` the
   number comes from.
3. Implement `prober.py` and integrate it into `train.py` exactly as specified below.

Step 1 — Read the codebase
Read `train.py` carefully. Understand:
- The task type (classification / regression / forecasting / detection / etc.)
- The training loop structure and where per-epoch (or per-validation-step) state is available
- What variables, objects, or hooks are accessible at each stage

Step 2 — Decide the metric and its direction
Apply the tiered METRIC RULE above. Tier 1 first: if `train.py` already computes an eval / selection metric, that is
the tracked metric, with the same direction `train.py` treats it as (whatever it maximises is higher_is_better,
whatever it minimises is lower_is_better). Only if there is no such metric at all do you go to Tier 2 (AUPRC,
higher_is_better) and then Tier 3. Note the direction — "higher_is_better" (e.g. AUPRC, ROC-AUC, accuracy, F1, mAP)
or "lower_is_better" (e.g. validation loss, RMSE, MAE, perplexity). Do NOT pick a threshold; do NOT decide a target.
The probe just monitors the metric.

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
         "conclusion": "string",
         "original_train_metric_0": {"name": "string", "value": float, "direction": "higher_is_better" | "lower_is_better"},
         "original_train_metric_1": {"name": "string", "value": float, "direction": "higher_is_better" | "lower_is_better"}
     }

  C. Create the metric directory if it does not exist. The save is mandatory — `conclude()` must not return without writing this file.

  D. ORIGINAL TRAIN METRIC ANCHOR (mandatory). The `original_train_metric_*` entries above are the ANCHOR: the
     model's OWN objective, recorded every round so the orchestrator can auto-revert any round that improves the
     probe metric by sacrificing what `train.py` was actually built to do. Record exactly two:
       - `original_train_metric_0` — `train.py`'s own training loss (name it as `train.py` names it,
         direction "lower_is_better").
       - `original_train_metric_1` — `train.py`'s own primary eval / selection metric, with the direction
         `train.py` treats it as.
     Use the final-epoch value for each (or the last value recorded, if you capture a series). If the tracked probe
     metric IS that same eval metric (the Tier-1 case, which is expected and fine), record it here anyway — the
     anchor and the probe metric are allowed to coincide. If `train.py` has only a training loss and no eval metric,
     record `original_train_metric_0` only and omit `original_train_metric_1`. Pass the anchor values into the probe
     the same way you pass the tracked metric (e.g. an extra `record(...)` argument, or a small
     `record_anchor(...)` helper you define) — never recompute them your own way.

  Important: in this auto-research mode there is NO plot output. Do not generate any chart, PDF, image, or plot file. The probe writes only the metric JSON. Do not import plotly / matplotlib for plotting purposes. Do NOT include `threshold`, `status`, `PASS`, or `FAIL` fields in the JSON or anywhere in the prober logic.

Step 4 — Integrate into `train.py`
Modify `train.py` to:
- Import `record` and `conclude` from `prober.py`
- Call `record(epoch, ...)` inside the training loop at the appropriate validation point each epoch
- Call `conclude()` (with NO arguments) exactly once after the training loop ends
- Mark the anchor: put a `# ANCHOR: original train metric` comment on the line where `train.py` computes its own
  training loss, and another on the line where it computes its primary eval / selection metric, and feed both
  values through to the probe so `conclude()` can write the `original_train_metric_*` entries. These markers are
  what every later improvement round is told to preserve — place them precisely.
- Do not alter any training logic — only add the import, the two calls, the anchor markers, and the minimal
  plumbing that carries the metric and anchor values into the probe

Constraints:
- Tier 1 of the METRIC RULE wins unless it genuinely does not apply: track the number `train.py` already improves
- Save metric output to the same `.agent_probe/metric/probe_result_N.json` location used by the rest of the pipeline — do not invent a new path
- Do not write a plot of any kind
- Do not introduce any threshold / target / pass-fail logic
- Do not modify files other than `prober.py` and the integration points in `train.py`
"""


# ── DISABLED: comment-driven auto-research iteration engine ──────────────────
# Superseded. Auto-research now improves the subject through the SAME fix-plan
# machinery auto-probe uses (PROMPT_NINE generate → PROMPT_TEN confidence →
# PROMPT_ELEVEN apply, with user hints and the anchor guard), so the mode no
# longer needs the commentor (PROMPT_SIX) to seed `# potential_improvement_*`
# markers nor this prompt to action them. Nothing imports this constant; it is
# kept verbatim for reference / quick revert, in the same spirit as the other
# disabled mechanisms in this repo. Do NOT re-wire it without also re-enabling
# the commentor in `auto_research_setup`. See AUTO_RESEARCH_CHANGE_REPORT.md.
PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT = """
You are an expert ML/DL optimization engineer working in auto-research mode. `train.py` already contains 10 inline comments labeled `# potential_improvement_1:` through `# potential_improvement_10:` — each one marks a specific place where a targeted change could move the probe metric in the better direction.

Your job for THIS iteration is simple and strict: pick exactly ONE of those comments and apply only the change it suggests. Leave everything else in the file untouched.

Important: this is a PERFORMANCE-MONITORING probe. There is NO threshold and NO pass/fail. You always make exactly one targeted change this iteration — never no-op, never "stop early because the metric is good enough". The orchestrator decides when iterations end, not you.

Revert-on-regression is handled by the orchestrator, NOT by you. After this iteration runs, the orchestrator compares the new probe_result's `tail_mean` to the running best (respecting `direction`); if your change did not improve the best, the orchestrator rewinds `train.py` to the pre-iteration snapshot and the per-run chart simply holds at the prior best. You do not need to inspect snapshots, compare results, or roll back `train.py` yourself — make one meaningful change every round, and accept that some rounds will be reverted by the orchestrator. That is expected.

Epoch budget — strict, always applies: you may NOT increase the number of training epochs as your single change for this round, and you may not raise the existing epoch cap past 50 even as a side-effect of another edit. Epochs are an expensive, indiscriminate lever and do not address the underlying issue. If a `potential_improvement_*` comment explicitly suggests raising epochs (or `n_estimators` past a clearly reasonable ceiling for the algorithm — e.g. > 2000 for GBDT — used as a stand-in for "train longer"), SKIP that comment and pick a different one. Early-stopping toggles or patience changes that leave `max_epochs` untouched are fine. Touching epoch-adjacent code without changing the cap is fine. Bumping `epochs` from any value up to a number larger than 50, or by more than ~50% in a single round, is not — even if a comment encourages it.

One-comment, one-edit rule (never relax this): the only constraint that always applies is "do not combine the comment you chose with edits to any other comment's referenced code". You touch ONE comment's referenced code per iteration. Nothing else.

Step size depends on the regime — read the latest probe_result JSON before deciding:

**Regime A: metric is at or near random baseline** (e.g. ROC-AUC ≈ 0.50, accuracy ≈ majority-class prior, F1 ≈ 0, RMSE ≈ a naive constant-predictor's RMSE). The model is in a DEGENERATE state — at least one parameter is so far outside reasonable territory that no learning is happening. Several parameters are likely crippling at once, so a small partial step on any single one will fail to break out and the metric stays flat for the next round too (a wasted iteration).
- Apply the comment's named target value in FULL. If the comment names a specific value, move to that value. If the comment names a range (e.g. "raise NUM_LEAVES to 31-127"), move to the value most likely to escape the degenerate regime (typically the more impactful end for the current metric direction, e.g. higher capacity if the model is too weak).
- Do not water the comment down. Half-measures in this regime are the failure mode.

**Regime B: metric has clearly moved off baseline and is climbing or has plateaued above baseline.** The model is learning; the goal is now refinement.
- Take a moderate step toward the comment's target — roughly 2x-5x the current value for a hyperparameter, or about half the distance toward the comment's named value if it specifies one.
- Aggressive moves risk overshooting and forcing a revert next round.

**Recent-progress check — modulate the step magnitude inside the regime.** Smooth, traceable round-to-round increments are a goal of auto-research mode: they make it easy to attribute the metric movement to a specific change. Wild swings obscure causality even when they happen to land in a good place. So before you commit to a magnitude, read recent history:

- Count the `.agent_probe/metric/probe_result_*.json` files. If there are fewer than two, **skip this entire check** — there is no prior round to compare against. Fall back to the Regime A / Regime B step size on first principles and move on.
- Otherwise, read the two latest `probe_result_*.json` files and compute `delta = tail_mean(latest) − tail_mean(previous)`, signed so that positive means "better" given `direction`.
- Also read the latest `change_log_*.txt` to see how aggressive the prior round's edit was (a small numeric bump vs. a categorical switch vs. a structural change to a block of code). If no `change_log_*.txt` files exist either, treat the prior edit as "unknown / assume small".

Then adapt:
- **Prior delta is negligible (roughly ≤ 1% of |current best|, zero, or negative) AND the prior edit was already small:** the model didn't budge — try harder this round. Pick a comment that targets a different lever, or push the value further toward the comment's named target (still within Regime A/B bounds). The orchestrator will revert if it overshoots, so escalating here is safe.
- **Prior delta is negligible AND the prior edit was already large/structural:** the lever is wrong, not the magnitude. Switch to a different `potential_improvement_*` comment rather than doubling down on the same one.
- **Prior delta is clearly positive (visibly beyond noise — usually ≥ a few percent of |current best|):** stay incremental. Prefer the smaller end of Regime B's 2x-5x range, or pick a different comment to layer in modest extra progress. A big jump now risks overshooting and forcing an orchestrator revert that wipes out the round's information value.
- **In all cases, prefer the smallest edit that plausibly produces a meaningful delta.** "Smaller, smoother, traceable" beats "bigger, faster, lucky" — the chart needs to tell a coherent story round by round.

**Always-fully-applied changes (in BOTH regimes):**
- Boolean toggles (e.g. `INCLUDE_AUXILIARY = True`, `USE_STRATIFY = True`, `DUMMY_NA = True`). Flip them in one move.
- Categorical / discrete choices (e.g. `BOOSTING_TYPE='rf' → 'gbdt'`, switching optimizer, switching loss). Set them to the recommended option directly.
- Removing or disabling a clearly-broken cap or guard (e.g. `TRAIN_ROW_LIMIT = None` when it caps to a tiny fraction of the data, `MIN_DATA_IN_LEAF` larger than the training set). Set the value to the sane setting in one move; do not "partially uncap".
- Comically-bad current values that are degenerate by themselves (e.g. learning_rate ≥ 0.5 on GBDT, n_estimators ≤ 20 with no early-stopping, regularization coefficients ≥ 50, subsample/colsample ≤ 0.1). Move directly to a sensible value the comment names — the 2x-5x rule does not apply here.

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
After making your change, write a plain-text summary to `.agent_probe/change_log_K.txt`, where K is determined as follows:
- List the existing `.agent_probe/change_log_*.txt` files. Take the highest numeric suffix and add 1. If there are no existing change_log files, K = 1.
- K should equal the index of the `probe_result_*.json` file your training run is about to produce (they share the same round number).

Include in the change log:
- Which `potential_improvement_*` item you chose (by number)
- The exact change you made (one or two sentences naming the parameter, value, or pattern)
- One sentence on why this change is expected to move the metric in the better direction

Step 5 — Verify integration integrity
Before finishing, re-read the `record(...)` and `conclude(...)` call sites in `train.py` and confirm:
- Both calls are still present and unmodified
- The arguments passed to `record()` still resolve to existing variables with the correct types
- `conclude()` is still called with no arguments

Make exactly one targeted change now — no more, no less.
"""


# ── Auto-research addendum for the SHARED fix-plan prompts ───────────────────
# Appended (auto-research mode only) to PROMPT_NINE / PROMPT_TEN / PROMPT_ELEVEN
# — the auto-probe fix-plan prompts, which are shared verbatim and must not be
# edited. Those prompts assume a threshold-gated probe (`standard_threshold`,
# `acceptable_threshold`, `status`, a two-argument `conclude()`); auto-research
# has none of that. This block reconciles the difference at call time so the
# improvement loop itself stays byte-identical across both modes.
PROMPT_AUTO_RESEARCH_PATCH_FIX_PLAN_ADDENDUM = """

────────────────────────────────────────────────────────────────────────────
AUTO-RESEARCH MODE ADDENDUM — READ THIS LAST; IT OVERRIDES THE PROMPT ABOVE
WHEREVER THE TWO CONFLICT.

This run is in auto-research mode. The improvement loop is exactly the one
described above (candidate fix plans → confidence → apply exactly one plan →
change log → the orchestrator runs training and keeps or reverts the round).
Only the probe differs: it is a PERFORMANCE-MONITORING probe, so every
threshold / pass-fail notion in the prompt above does not exist here.

1. NO THRESHOLD, NO PASS/FAIL. There is no `standard_threshold`, no
   `acceptable_threshold` and no `status` field — not in `prober.py`, not in
   `.agent_probe/metric/probe_result_*.json`. Ignore every instruction above
   that tells you to read, compare against, or report them; do not go looking
   for them, and do not add them. "The probe is failing" above simply means
   "the tracked metric has room to improve" — which is always true here.

2. THE OBJECTIVE IS UNBOUNDED IMPROVEMENT. Push the tracked metric further in
   its better direction (`direction` in the probe_result), round after round.
   Never conclude the metric is "good enough", never propose a no-op plan, and
   never stop early. The orchestrator decides when the rounds end, not you.

3. `conclude()` TAKES NO ARGUMENTS. Where the prompt above says to keep
   `conclude(standard_threshold, acceptable_threshold)` intact with both
   values passed, read it as: keep the `conclude()` call site exactly as it
   is — a no-argument call. Do not add arguments to it. The `record(...)`
   call site rule is unchanged: leave it intact and keep its arguments
   resolving to existing variables of the right type.

4. `prober.py` IS STILL COMPLETELY OFF-LIMITS, and the probe's metric
   definition, aggregation and operating point are still FROZEN — including
   any probe-facing constant referenced from `train.py`. The tracked metric is
   usually the metric `train.py` itself already computes and selects on; you
   improve it by genuinely improving the model, never by redefining,
   re-aggregating or re-thresholding how it is measured.

5. THE ANCHOR AND `user_analyze()` RULES STAY FULLY IN FORCE. Keep the
   `# ANCHOR: original train metric` markers and the `original_train_metric*`
   recording intact and accurate, and keep `user_analyze()` untouched. A round
   that degrades an anchor by more than 20% from the round-1 baseline is still
   auto-reverted by the orchestrator and still wastes the round.

6. NO PLOT OUTPUT. This mode writes only the metric JSON. Do not add any
   chart / PDF / image generation to `prober.py` or `train.py`.

7. EPOCH BUDGET — STRICT, ALWAYS APPLIES. Auto-research runs many rounds, so
   "just train longer" is an expensive, indiscriminate lever that hides the
   real bottleneck. A plan may NOT increase the number of training epochs as
   its change, and may not raise an existing epoch cap past 50 even as a side
   effect. Do not propose, score highly, or apply such a plan; the same bar
   applies to `n_estimators` pushed past a clearly reasonable ceiling for the
   algorithm (e.g. > 2000 for GBDT) as a stand-in for "train longer".
   Early-stopping toggles and patience changes that leave the epoch cap
   untouched are fine, as is touching epoch-adjacent code without raising the
   cap. Bumping an epoch count above 50, or by more than ~50% in one round,
   is not.
────────────────────────────────────────────────────────────────────────────
"""
