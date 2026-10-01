# Auto-research ⇄ auto-probe: one shared improvement loop

Branch: `auto-research-unified-improve` (off `selfcontained-case-train`).
Nothing was pushed; no other branch was touched.

---

## 1. How AutoProbe works (short)

A **workspace** is any folder containing `train.py` (`python train.py`, no
args). The orchestrator (`pipeline/`) drives `claude` subprocesses
(`pipeline/llm.py`: `nlp_call` = no tools / JSON out, `agent_call` = full tools
inside the workspace) through four stages, fronted by FastAPI (`server/app.py`)
and a Next.js UI (`web/`). Per-run state lives in `response/<run_id>/stage.json`
(`pipeline/state.py`); `train.py` is versioned in a private bare git repo at
`<ws>/.agent_probe/snapshot.git` (`pipeline/snapshot_git.py`) with tags
`baseline`, `pre-iter`, `round-N-post`.

**Auto probe** (the regular mode):

1. **Stage 1** — user context → 10 probe designs + confidence (PROMPT_ONE/TWO),
   user picks one.
2. **Stage 2** — 3 dev plans + practicality scores (PROMPT_THREE/FOUR), user
   picks one.
3. **Stage 3** — agent writes `prober.py` and integrates `train.py`
   (PROMPT_FIVE); tag `pre-iter`; run training (crashes → PROMPT_EIGHT, ≤5
   retries); tag `round-1-post`; record round 1.
4. **Stage 4** — the improvement loop. Each round: 3 candidate **fix plans**
   (PROMPT_NINE, plus an optional **user hint** appended verbatim) → supervisor
   confidence (PROMPT_TEN) → apply exactly one plan + change log
   (PROMPT_ELEVEN) → training → the **orchestrator** decides keep/revert
   (`_maybe_revert_on_regression`: this round's `tail_mean` vs the previous
   round's, plus the **anchor guard** — a >20 % collapse of `train.py`'s own
   `original_train_metric*` auto-reverts the round and leaves a one-shot
   warning for the next agent). Entry points: `auto_fix_loop` (auto-pilot,
   auto-picks the highest-confidence plan, stops at PASS / best-effort /
   stagnant) or the manual `generate_fix_plans` → `select_and_apply_fix_plan`.

Decisions key on `tail_mean` (mean of the last 5 epochs), computed by the
orchestrator, never trusted from the prober. `iterate_once` is a disabled stale
engine; the CLI's Stage 4 and `/stage4/iterate` are disabled with it.

## 2. What the old auto research did

Toggling auto-research at Stage 1 ran `auto_research_setup`, which:

- had the agent pick a metric on its own judgement (prompt rule: **AUPRC
  first**, otherwise "a standard metric for the task type"), write `prober.py`
  (no threshold, no plot) and integrate `train.py`;
- ran training, then ran the **improve-commentor** (PROMPT_SIX) to seed 10
  `# potential_improvement_1..10:` comments into `train.py`, then ran training
  again;
- **deleted** both validation runs' artifacts and cleared the iteration ledger
  so the first batch round would read as "1st run".

Stage 4 then used a *separate* engine: `auto_research_iterate_batch(count)`
looped `count` rounds of `PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT`
("pick exactly ONE of those 10 comments and apply it"), comparing each round's
**`last_epoch`** against a **running best** and rewinding `train.py` when it
didn't beat that best. There were no fix plans, no confidence scoring, no user
hint, and `generate_fix_plans` / `select_and_apply_fix_plan` / `auto_fix_loop`
all raised `ValueError` for auto-research runs.

## 3. What changed

### `hard_prompt/auto_research_prompt_patch.py` (auto-research-only prompts)

- **Setup prompt — new tiered METRIC RULE.** Tier 1 (the default, needs a
  stated reason to leave): track the metric `train.py` **already** computes and
  improves — the value it selects / early-stops / best-checkpoints on, else its
  headline per-epoch eval number, else the number it writes to a results file.
  The already-computed number is passed straight through to `record()`; the
  agent must not recompute, redefine, re-aggregate or re-threshold it. Tier 2 is
  the old AUPRC-first rule, now reachable only when `train.py` evaluates
  nothing. Tier 3 is the old "standard metric for the task type". The prober's
  docstring must state which tier was used and, for Tier 1, which line of
  `train.py` the number comes from.
- **Setup prompt — anchor + protected code.** It now opens with the same
  `user_analyze()` hard rule the stage-4 prompts carry, requires
  `# ANCHOR: original train metric` markers on `train.py`'s own training loss
  and primary eval metric, and requires `original_train_metric_0/1` in every
  `probe_result_N.json`. *Why:* the shared fix-plan prompts instruct the agent
  to preserve exactly those markers and the orchestrator's anchor guard is a
  no-op without them — auto-research previously had neither, so the shared
  prompts would have referred to things that did not exist.
- **New `PROMPT_AUTO_RESEARCH_PATCH_FIX_PLAN_ADDENDUM`.** Appended (auto-research
  only) after PROMPT_NINE / PROMPT_TEN / PROMPT_ELEVEN. It declares the mode —
  which is what the conditional line now in those prompts keys on (§8) — and
  carries the instructions that have no home in a shared prompt: the objective
  is unbounded improvement (never stop early, never no-op); `prober.py` and the
  frozen operating point are still off-limits; the anchor and `user_analyze()`
  rules stay in force; no plot output; and the auto-research **epoch budget** (a
  fix plan may not raise epochs, nor an epoch cap past 50). It also restates the
  no-threshold / no-argument-`conclude()` facts, since it is the block the
  inline line points at.
- `PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT` kept verbatim under a
  `# ── DISABLED` banner (repo convention), with no importer.

### `hard_prompt/agent_fix_plan_gen.py`, `hard_prompt/agent_fix_plan_apply.py`

- **One conditional line each** (PROMPT_NINE, PROMPT_ELEVEN), added under
  explicit permission from the owner, stating that threshold instructions and
  the two-argument `conclude()` don't apply when the auto-research addendum is
  present. Nothing else in either prompt was reworded. PROMPT_TEN and
  PROMPT_EIGHT were checked and left alone — neither assumes a threshold. Full
  detail, placement rationale and the inertness argument: **§8**.

### `hard_prompt/agent_improve_commentor.py`

- `PROMPT_SIX` left **byte-identical**, under a `# ── DISABLED` banner
  explaining it now has no caller.

### `pipeline/stages.py`

- `auto_research_setup` rewritten to mirror the tail of `implement()` step for
  step: agent → tag `pre-iter` → **one** training run (`probe_result_1`) → tag
  `round-1-post` → record round 1. The commentor call, the second validation
  run and the artifact/ledger wipe are gone. *Why:* the shared fix-plan flow
  requires `prober.py` plus ≥1 iteration row, and the anchor guard baselines on
  `probe_result_1`; after this, an auto-research run at stage 4 is
  indistinguishable from an auto-probe run at stage 4.
- The three `if debug_flags["auto_research"]: raise ValueError(...)` guards in
  `generate_fix_plans`, `auto_fix_loop` and `select_and_apply_fix_plan` are
  removed — those functions are now the shared path.
- `_mode_addendum(state)` → `""` for auto probe, the addendum for auto
  research. Appended to the three shared prompts — and it doubles as the flag
  the shared prompts' conditional line tests for (§8). `_hint_block(hint)`
  factored out of `generate_fix_plans` so the hint is formatted identically
  everywhere.
- `_do_auto_fix_round(state, hint=None)` — now **the** improvement round for
  both modes. Gained the optional hint (default `None` reproduces the previous
  prompt byte-for-byte) and returns its `IterationRecord`.
- `auto_research_iterate_batch(state, count, hint=None)` rewritten to a loop of
  `_do_auto_fix_round`, tracking `auto_research_runs_completed` /
  `_target_runs` for the UI. All of the old bespoke keep/revert, best-tracking
  and artifact-stamping code is gone.
- `_set_auto_research_best` / `_track_auto_research_best` — maintain the
  **display-only** running best and stamp it on each stored row as
  `best_value` (the monotonic per-run chart), and null the threshold fields a
  stale prober might have written. No-ops in auto-probe mode.
- `_backfill_auto_research_baseline` — a run parked at stage 4 by the *old*
  setup has an empty ledger; it is reconstructed from the on-disk
  `probe_result_*` (and the missing `round-N-post` tag is created) instead of
  the run becoming unusable.
- `_annotate_metric_history` commented out under a DISABLED banner (it
  described the retired best-based comparison);
  `_read_best_stats_and_direction` kept but documented as unused.

### `server/app.py`

- `AutoResearchIterateBody` gained optional `hint`; the route forwards it. A
  legacy `{"count": N}` body still validates. The `/stage1/auto-research`
  docstring no longer mentions the commentor. No auto-probe route changed.

### `web/` (UI)

- `api.ts`: `autoResearchIterate(runId, count, hint?)`.
- `Stage4.tsx`: in auto-research mode the hint textarea is always shown when no
  plans are open (it feeds both the batch and a manual round), a
  **"Generate fixing plans"** button exposes the manual fix-plan round, and the
  fix-plan cards panel is no longer gated to auto-probe. Starting a batch clears
  `fixingMode` so stale plans can't flicker mid-round. Subtitle rewritten.
  The auto-probe branch is untouched (`handleGenerateFixPlans` sets a flag that
  was already true on every auto-probe path that can reach it).
- `Stage1.tsx`: auto-research card copy now describes metric reuse.
- `ui.tsx`: label for the new `auto-research-setup-test-run` action; the two
  legacy `auto-research-*` labels kept so old runs still read well.

### Docs / tests

- `README.md`: auto-research section, diagram, walkthrough, artifact table,
  prompt tree and design-decision bullets rewritten; the demo video is flagged
  as predating this change.
- `CLAUDE.md`: live Stage-4 function list, the "how to run" routes, and a note
  that PROMPT_SIX and the comment-driven iteration prompt are callerless.
- `tests/test_unified_stage4.py`: new offline harness (see §7).

## 4. How auto research now reuses the auto-probe improvement loop

After setup, an auto-research run holds exactly what auto-probe holds entering
stage 4: `prober.py`, `probe_result_1.json`, tags `pre-iter` + `round-1-post`,
one iteration row. From there **both modes call the same functions**:

| | auto probe | auto research |
|---|---|---|
| Stage 1 probe idea | PROMPT_ONE/TWO + user pick | **skipped** |
| Stage 2 dev plan | PROMPT_THREE/FOUR + user pick | **skipped** |
| Probe implementation | `implement()` (PROMPT_FIVE) | `auto_research_setup()` |
| Fix-plan generation | `generate_fix_plans` (PROMPT_NINE) | same function |
| User hint | appended to PROMPT_NINE | same, plus available to the auto-pilot |
| Confidence scoring | PROMPT_TEN | same |
| Plan application | PROMPT_ELEVEN | same |
| Change log | `.agent_probe/change_log_N.txt` | same |
| Crash recovery | PROMPT_EIGHT, ≤5 retries | same |
| Keep / revert | `_maybe_revert_on_regression` (round N vs N−1, `tail_mean`) | same |
| Anchor guard | `_anchor_guard_breach` + one-shot warning | same |
| Auto-pilot | `auto_fix_loop` → stops at PASS / best-effort / stagnant | `auto_research_iterate_batch(count)` → stops after `count` rounds |
| Manual round | `fix-plans/generate` → `fix-plans/select` | same routes |

The single remaining difference inside stage 4 is the **stopping rule**, and it
is forced: auto-research has no threshold, so there is no PASS, and
`_at_terminal_state` is unconditionally terminal past 4 rows — using it would
cap auto-research at 3 rounds and ignore the user's count. The round body is
byte-identical.

## 5. How the `train.py` metric is preferred

Tier 1 of the setup prompt's METRIC RULE (§3) makes "the metric `train.py` is
already improving" the default, with AUPRC demoted to a fallback for scripts
that evaluate nothing. The rule names the search order (selection/early-stopping
value → headline printed eval → results file), forbids recomputation ("pass the
already-computed number straight through to `record()`"), forbids redefinition /
re-aggregation / re-thresholding, and requires the prober docstring to state the
tier and the source line. The addendum reinforces it at every improvement round
("the tracked metric is usually the metric `train.py` itself already computes
and selects on; you improve it by genuinely improving the model").

Note the healthy side effect: when Tier 1 applies, the probe metric and the
anchor coincide, so "improve the probe metric without sacrificing what the
script was built to do" becomes tautological rather than adversarial. The prompt
says so explicitly, so the agent doesn't try to pick a *different* metric to
keep them separate.

## 6. How the commentor is disabled

Four things, all of them inert rather than deleted:

1. `auto_research_setup` no longer calls `agent_call(PROMPT_SIX…)` (nor the
   second validation run it required).
2. `pipeline/stages.py` no longer imports `hard_prompt.agent_improve_commentor`.
3. `PROMPT_SIX` sits under a `# ── DISABLED` banner, prompt text unedited.
4. `PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT` — the only consumer of the
   `# potential_improvement_*` markers — is disabled and unimported too.

The test harness asserts both the static facts (no import, no interpolation) and
the dynamic one (the stub raises if the commentor prompt is ever sent;
auto-research setup makes exactly **one** agent call).

## 7. Tests / verification done

All offline — no `claude` subprocess, no GPU, no LLM spend.

- **`venv/bin/python tests/test_unified_stage4.py` → 82 checks, 0 failures.**
  It runs the real `pipeline.stages` code against a temp workspace with a
  20-line fake `train.py` (writes a `probe_result` JSON, carries an anchor and a
  `VALUE` constant a "fix plan" edits) and a scripted stand-in for
  `agent_call`. So `run_training_with_autofix`, output normalization,
  snapshot-git tagging, the keep/revert decision and the anchor guard all
  execute for real. Covered:
  - setup parks at stage 4 with one training run, `probe_result_1`, both tags,
    one iteration row, a seeded best — and **one** agent call (no commentor);
  - a 2-round batch issues exactly generate → confidence → apply twice, every
    shared prompt carries the addendum, the hint reaches the generator each
    round, rounds 2–3 produce metrics and `round-N-post` tags, the auto-pilot
    picks the highest-confidence plan, improving rounds are kept, the metric
    advances 0.50 → 0.54, `best_value` is monotonic, no threshold fields leak;
  - a regressing round is reverted and `train.py` is rewound to the previous
    round's snapshot;
  - **auto probe sees nothing but the shared prompt** —
    `== f"{PROMPT_NINE}{round}"`, `== f"{PROMPT_TEN}{round}"`,
    `== f"{PROMPT_ELEVEN}\n\n{json}"`, the hinted generate prompt
    `== f"{PROMPT_NINE}{round}\n\nUser hint:\n…\n"`, and no addendum anywhere.
    The conditional AUTO-RESEARCH MODE line (§8) is asserted *present* in the
    prompt and the block it names asserted *absent*, i.e. provably inert;
  - auto-probe's auto-pilot still runs 3 fix rounds, stops at 4 rows, parks at
    `phase=ready`, is re-enterable, and gets no `best_value` bookkeeping;
  - the manual fix-plan round works in auto-research mode (it used to raise);
  - a legacy auto-research run with an empty ledger recovers a baseline row;
  - static guards: no commentor import/send, no comment-driven prompt use, no
    mode guards left in the three shared functions, Tier 1 precedes Tier 2 in
    the setup prompt, the anchor is wired, `iterate_once` still raises, and
    `_at_terminal_state` still returns False/True/True for 3-FAIL / 4-FAIL /
    PASS;
  - the conditional line: present in PROMPT_NINE and PROMPT_ELEVEN exactly
    once each, absent from PROMPT_TEN, phrased as a conditional that names the
    addendum block and states the "if absent" branch, the addendum carries the
    matching banner, and the trimmed addendum still states both facts.
- **Live API smoke** (`AUTOPROBE_ROUTER=off`, port 8799, no LLM reachable):
  `/api/health` ok; the OpenAPI schema shows `hint` as optional on
  `AutoResearchIterateBody`; `count=0` and `count=101` → 400; a valid body and
  a legacy `{"count":2}` body both reach the handler (500 only because the run
  id doesn't exist — nothing launched); the stale `/stage4/iterate` route is
  still 404.
- `venv/bin/python -m py_compile` on every touched module; `server.app` imports;
  `npx tsc --noEmit` clean; `python main.py list` still works (CLI Stage 4 still
  `SystemExit`s as before).

**Not done:** no live end-to-end run with a real agent. The ccr gateway is not
serving on this box (`:3456` health = 000) and starting it plus running
opus/deepseek agents over a real case with GPU training is neither cheap nor
something to drive headlessly — CLAUDE.md says auto-probe/auto-research must be
driven through `make api` + `make web`. So the *orchestration* is verified and
the *prompt compliance* is not. First live check worth doing, on a small case:
auto-research setup → confirm `prober.py` tracks the metric `train.py` already
reports and records `original_train_metric*` → one batch round → confirm
`fix_plans_2.json` exists with confidences and a change log.

## 8. Shared-prompt edits (PERMISSION GRANTED, now applied)

The owner granted permission for one minimal conditional line: *"just add a
line that unless we are doing auto research then we don't have threshold so
you can ignore threshold related prompt in any confusing prompt."* Applied as
**one line, verbatim identical in both prompts**, and nothing else reworded:

```
AUTO-RESEARCH MODE (conditional — read this before anything below): if an
"AUTO-RESEARCH MODE ADDENDUM" block appears at the END of this prompt, this
run's probe is a performance monitor with NO thresholds — `standard_threshold`,
`acceptable_threshold`, `status`, PASS and FAIL do not exist anywhere, so
ignore every threshold-related instruction below, and `conclude()` is called
with NO arguments (leave that call site exactly as you find it). Everything
else below still applies unchanged. If that block is absent, this is a normal
threshold-gated probe and every instruction below applies as written.
```

(Shown wrapped; it is a single line in the source.)

- **`hard_prompt/agent_fix_plan_gen.py` → PROMPT_NINE** — inserted as the
  *first* line, because the prompt's own opening sentence ("The current probe
  is failing — the metric has not crossed `standard_threshold`") is the
  confusing claim, so the caveat has to be read first. The threshold bullets in
  its Step 1 / Step 2 are covered by the same line and were left untouched.
- **`hard_prompt/agent_fix_plan_apply.py` → PROMPT_ELEVEN** — inserted after
  the three HARD RULE paragraphs (which are mode-agnostic and belong at the
  top) and immediately before the body, which is where the two
  `conclude(standard_threshold, acceptable_threshold)` instructions live
  (Step 2's "do not modify" bullet and Step 4's verification bullet). Both
  bullets were left untouched; the line covers them.
- **`hard_prompt/agent_fix_plan_gen.py` → PROMPT_TEN — deliberately NOT
  touched.** Checked: it contains no occurrence of `threshold`, `PASS`, `FAIL`
  or `conclude`, so it makes no assumption to correct. Adding the line there
  would have been noise.
- **`hard_prompt/agent_exception_catcher.py` → PROMPT_EIGHT — NOT touched.**
  Checked: its only `conclude()` mention is mode-agnostic ("wrong arguments
  passed to `record()` or `conclude()`" → fix the call site), which is correct
  advice for a zero-argument call too.
- **`hard_prompt/agent_iterat_improver.py` → PROMPT_SEVEN — NOT touched.** It
  belongs to the disabled `iterate_once` engine and is never sent. (Still
  imported under `# noqa: F401` purely to keep the reference alive — say the
  word and I'll drop the import.)

**Auto-probe behaviour is unchanged** because the line is conditional on a
block auto-probe never receives: `_mode_addendum()` returns `""` outside
auto-research, so the named "AUTO-RESEARCH MODE ADDENDUM" block is absent and
the line's own final clause applies — "every instruction below applies as
written". The test suite asserts exactly this (§7).

**Addendum trimmed in exchange.** Its two retraction items (item 1 "NO
THRESHOLD, NO PASS/FAIL" and item 3 "`conclude()` TAKES NO ARGUMENTS") are now
redundant with the inline line, so they were merged into a single short item
that points back at it; items 2/4/5/6/7 renumbered to 2/3/4/5/6. The addendum
still *states* both facts (the tests pin that) because it is also the flag the
inline line keys on — it has to announce the mode. Merging the two items saved
~110 characters; two additions put it back and a little more (a cross-reference
in the header, and a reminder that the round index `K` given at the end of
PROMPT_NINE is still the integer immediately above the block), so net size went
3523 → 3559 characters. The win is placement, not length: the correction the
agent most needs now arrives before the claim it corrects.

## 9. Open risks

1. **One-step ratchet over long batches.** Auto-probe's keep/revert compares
   round N to round **N−1**, not to the running best. A regressing round is
   reverted in `train.py` but its (worse) metric is still the row the next round
   is judged against, so round N+1 can be "kept" while sitting below an earlier
   peak. Over auto-probe's ≤3 rounds this barely matters; over a 50-round
   auto-research batch it can drift away from the best version seen. This is a
   direct consequence of "exact same logic as auto probe", and the old
   auto-research code did *not* have it. If you want the old guarantee back, the
   change is small and local: in `_maybe_revert_on_regression`, compare against
   `state.record.auto_research_best_value` instead of `iters[-1]` when
   `debug_flags["auto_research"]` is set. I did not do it because it would make
   the two modes differ in the one place you asked them not to — and the owner
   has since confirmed they want auto-probe's N vs N−1 rule kept as is. Note
   the precise consequence: the rule does **not** guarantee the kept `train.py`
   never sits below an earlier peak. After any reverted round, the comparison
   baseline is the *discarded* code's metric, so the next round only has to
   beat the failed attempt. Worked example (higher_is_better): round 1 = 0.70
   (train.py = A, tag round-1-post = A); round 2 edits A→B, measures 0.60,
   reverted, train.py back to A, but the ledger row for round 2 is 0.60; round
   3 edits A→C, measures 0.65, compared against 0.60 → **kept**, so train.py is
   now C at 0.65, below round 1's 0.70, and every later round is judged against
   0.65. The per-run chart will still look monotonic because it plots
   `best_value` (the running best ever seen), not the metric of the code
   currently on disk. There is no revert *tolerance* contributing to this:
   `_better_strict` demands strict improvement at 4-decimal precision, so ties
   revert.
2. **Cost per round went up ~3×.** A round used to be 1 agent call + 1 training
   run; it is now 3 agent calls (generate / score / apply) + 1 training run. A
   100-round batch is a materially bigger bill than before.
3. **A mid-batch failure aborts the batch.** If fix-plan generation produces no
   file, or training can't be fixed in 5 retries, the whole batch stops and the
   error surfaces as `last_error` (the old loop `continue`d past a round that
   produced no metric). This matches `auto_fix_loop`; completed rounds are
   durable, so re-starting a batch resumes cleanly.
4. **Prompt compliance is unverified** (see §7). Three specific things to watch
   on the first live run: whether the agent honours the conditional
   AUTO-RESEARCH MODE line (i.e. does not hunt for thresholds and does not
   "restore" arguments to `conclude()`); whether the setup agent really reuses
   `train.py`'s metric rather than inventing one; and whether it places the
   anchor markers precisely enough that `original_train_metric*` is recorded
   every round (the anchor guard silently no-ops if either side of the
   comparison lacks anchors).
5. **The addendum is long** (~3.5 kB) and sits at the end of three already-long
   prompts. The worst part of that — retracting the prompt's own opening claim
   4 000 characters later — is now fixed at source by the conditional line
   (§8), which is the first thing the agent reads in PROMPT_NINE. What remains
   is bulk: the addendum also lands *after* PROMPT_NINE's trailing round index
   `K`, so it now carries an explicit reminder that `K` is the integer
   immediately above it.
6. **Auto-research now writes `fix_plans_N.json` into the workspace and the run
   folder.** `revert_to` already purges both, so nothing leaks, but the
   per-round artifact set for an auto-research run is larger than it was.
7. **The `autoresearch.mp4` demo is now stale** — it shows the comment-driven
   engine. Flagged in the README rather than re-recorded.
