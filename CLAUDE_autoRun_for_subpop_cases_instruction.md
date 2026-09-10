# Auto-running a SubpopBench case, end to end

Operational runbook for an agent asked to *"auto-run AutoProbe on `cases/<Case>` like a
human user."* It assumes you have already read [`CLAUDE.md`](CLAUDE.md) (which tells you
the stale Stage-4 CLI engine is off-limits) and [`README.md`](README.md).


**Read [§1 Your role](#1-your-role-supervisor-not-operator) before you touch anything.** It
is the operating contract, and it overrides any habit you have about starting servers,
tailing logs, or "fixing" a stuck run. Then read
[§9 Known hazards](#9-known-hazards-learned-the-hard-way) — several of them cost hours.

---

## 0. What you are actually trying to do

Not "make a number go up." You are testing whether the pipeline can, given a project
description, **detect** a subpopulation-shift failure and then **fix it**, with the fix
verified by the benchmark's own unmodified scorer.

That framing has consequences:

- The fix must be a real training-side change. Moving a metric by editing the scorer,
  the dataset, or the probe is not a result, it is a bug.
- Prefer a fix whose mechanism you can name. "Randomise the spurious colour channel so
  the model must use shape" is a result. "Train fewer epochs" is a weaker one — legitimate
  (it is early stopping) but it says only *don't overfit*.
- **Do not look up the paper's published numbers** if the user intends to validate against
  the benchmark afterwards. Knowing the target contaminates the experiment and makes the
  later comparison circular. Hinting from *your own probe's measured per-epoch data* is
  fine and expected — that is what the probe is for.

---

## 1. Your role: supervisor, not operator

You drive AutoProbe through its REST API at `http://127.0.0.1:8765` and answer the
pipeline's user-facing prompts. **You do not edit the workspace, and you do not own any
long-running process.**

### 1.1 Process isolation — read first

A run can last many hours. Anything you start from a Bash tool call is a child of your
session and dies when that session ends or compacts. Everything long-lived goes in tmux.

**1. A tmux server must already be running. Never start one yourself.** Verify before
anything else:

```bash
tmux ls
```

If that errors, **STOP** and tell the user to run `tmux new-session -d -s scratch` first.
Do not work around it.

**2. Never run `make api`, `make web`, `python main.py`, or `python train.py` directly
from a Bash call.** Always wrap in tmux:

```bash
tmux new-session -d -s ap_api -c <REPO> 'source venv/bin/activate && make api 2>&1 | tee /tmp/ap_api.log'
tmux new-session -d -s ap_web -c <REPO> 'make web 2>&1 | tee /tmp/ap_web.log'
```

Environment does not persist between Bash calls — **activate the venv on the SAME command
line**, never in a prior call. (`-c <REPO>` also sets the working directory, so the old
`cd`-doesn't-persist workaround is unnecessary here.)

**3. Verify isolation once, and report the result:**

```bash
ps -o pid,ppid,cmd -p $(pgrep -f uvicorn | head -1)
```

The parent chain must lead to a tmux server with PPID 1, **never to a `claude` process**.
If it leads to claude, stop and say so.

**4. Any HTTP call that may block for more than a minute** (stage triggers, round batches)
goes through tmux too, not a foreground curl:

```bash
tmux new-session -d -s ap_trigger 'curl -sS -X POST <url> -d <payload> > /tmp/ap_trigger.log 2>&1'
```

Then poll. Do not hold a blocking connection open.

**5. Do not put `git` or `sleep` in any command you expect to outlive its timeout**, and do
not end commands with `; exit "${PIPESTATUS[0]}"`. Those forms are killed at the deadline
instead of backgrounded.

### 1.2 Context management

**6. NEVER tail, cat, read, or grep `response/<run_id>/agent.log`.** It is the full
streamed transcript of every model call and it will bury you within one round. If you need
a reason for a failure, read the **last 50 lines ONCE**, extract the answer, and do not
read it again.

**7. State comes from `response/<run_id>/stage.json`.** It is small and it is the single
source of truth for stage, phase, selections, and iteration history. Poll that.

> One caveat, verified in `server/app.py`: `busy` and `last_error` are **not** in
> `stage.json`. They live in server memory (`_last_error`) and are attached by
> `_state_payload` at request time. So when a phase change looks like an *unexplained*
> termination — `running` → `ready`/`input` with no new iteration row — make **one**
> `GET /api/runs/{run_id}` to read `last_error`. That is the only reliable way to see why a
> background action died, and it is far cheaper than opening `agent.log`.

**8. Watch by state change, not by stream.** Write a poller that compares `stage.json`
against its previous value and prints **ONE LINE only when phase changes**, then run it
under the Monitor tool with `persistent: true`. Silence while training, one event when the
pipeline needs you. **Never point Monitor at raw logs.** If you pipe through grep anywhere,
always use `grep --line-buffered`.

**9. Metrics:** read `.agent_probe/live/probe_live.json` or `probe_result_N.json` only when
a round ends, or when you need a number to make a decision. Never per epoch. Report deltas,
not full arrays.

**10. Keep your own output small.** After each round, append a two-line summary to
`/tmp/ap_supervisor_notes.md` (round N, metric, kept or reverted, next action) and rely on
that file rather than on remembering the transcript. **If context fills, the notes file is
what carries the run forward.**

### 1.3 Driving the pipeline

**11.** Fetch `http://127.0.0.1:8765/openapi.json` **ONCE** at start to learn the endpoints
and payload shapes. Do not guess URLs. The routes quoted later in this doc are a map, not a
substitute for that fetch. **Do not drive the Next.js UI on :3000** — that is for the human.

**12.** HTTP **409** means another stage holds the lock. It is normal, not an error. Back
off, wait, re-poll. Never retry in a tight loop.

**13.** When `stage.json` shows a waiting state (probe selection, dev plan, PASS/FAIL
dialog), make the choice, POST it, state in one sentence what you chose and why, and
continue.

### 1.4 Hard limits

**14. Never call `/api/cancel`. Never kill a tmux session, a uvicorn process, or a
`train.py` run. Never call `TaskStop` on anything the pipeline owns.** If you believe a run
should be stopped, say so and wait for the human.

> `TaskStop` on **your own** Monitor tasks is fine and sometimes necessary — two live
> monitors duplicate every event. The prohibition is about pipeline-owned processes.

**15. During a run, never edit `train.py`, `prober.py`, anything under `.agent_probe/`, or
the workspace.** Stage 3/4 agents own those files and the orchestrator snapshots and reverts
them; your edits corrupt that logic. **Read-only while a run is live.**

> After a run has finished — recovery, cleanup, repo maintenance — you *may* edit them, but
> **tell the user what you changed.** See [§10 Recovery](#10-recovery), which is scoped
> accordingly.

**16. If the pipeline crashes, do not fix it yourself.** It has up to 5 automatic retries
via `PROMPT_EIGHT`. Let them run. Report after the fifth.

**17. Do not modify router/auth environment or export `ANTHROPIC_MODEL`.** The pipeline
injects its own routing into its subprocesses; interfering breaks it.

### 1.5 Escalate to the human

- Verification in §1.1 step 3 shows a `claude` parent.
- Same crash 5 times, or a revert loop with no progress across 3+ rounds.
- NaN or diverging loss in the live metric.
- Any decision that would discard completed work.

---

## 2. Pre-flight

```bash
tmux ls                     # MUST already exist — see §1.1
cd /mnt/workspace/AutoProbe
make doctor                 # python / node / claude CLI / ccr / torch versions
git status --porcelain      # want a clean tree before you start
git rev-parse --abbrev-ref HEAD
ls /mnt/workspace/data/     # which datasets are actually present
```

If on the default branch, **branch first** — a run rewrites `train.py` many times.

### Never touch these

| Path | Why |
|---|---|
| `cases_copy/ or similar name including the word copy` | The pristine template set. Read it, restore *from* it, never edit it. |
| `subpopbench/dataset/datasets.py` | Defines the task, including the spurious correlation. Editing it changes the problem instead of solving it. |
| `subpopbench/utils/eval_helper.py` | `eval_metrics` is **the scorer**. Must stay definitionally identical to the benchmark and independently auditable. |
| `train.py`, `prober.py`, `.agent_probe/` | Owned by the Stage 3/4 agents and the snapshot/revert logic. Read-only **during a run** (rule 15). |
| `pipeline/`, `server/`, `web/` | Backend. Drive it, do not modify it. |
| `main.py` Stage-4 loop, `POST /stage4/iterate` | Stale engine. See `CLAUDE.md`. |

Everything an improvement round is *allowed* to change lives in the case's `train.py`,
which is self-contained (~2050 lines: hparams, optimizers, networks, joint-DRO, all 25
algorithms inlined). That is deliberate — see
[hazard 3](#hazard-3-revert-only-covers-trainpy). It is the **agents'** file to edit, not
yours.

### Know the runtime, but do not optimise for it

Runtime per training run scales with `N_STEPS`. Resolve it per case with
`get_dataset_class(DATASET).N_STEPS` (some classes inherit it, so read via the MRO, not
`vars()`):

| Cases | `N_STEPS` | relative | ≈ per run on one L4 |
|---|---|---|---|
| CMNIST, MetaShift, Waterbirds | 5 001 | 1× | ~45–50 min |
| ImagenetBG, MIMICNotes | 10 001 | 2× | ~1.5–2 h |
| CXRMultisite, CheXpertNoFinding, MIMICNoFinding | 20 001 | 4× | ~3–4 h |
| CelebA, CivilCommentsFine, MultiNLI, NICOpp | 30 001 | 6× | ~4–6 h |
| Entity13, Entity30, Living17, Nonliving26 | 60 001 | 12× | ~9–12 h |

The L4 column is a *reference point*, not a constraint. The intended deployment is **one VM
per case** on dedicated A100/H100-class hardware — and even there, **single training runs
above 5 hours are expected** on the larger cases. Wall-clock is not a reason to change the
experiment.

**Therefore: keep the paper-comparable step budget.** Do not lower `TOTAL_STEPS` or set
`MAX_EPOCHS` for turnaround — the whole point of these runs is a number that can be compared
against the published tables, and a shortened budget forfeits that. Reducing the budget is a
legitimate *research* move (early stopping, [§7](#ordering-the-attempts)) when the probe's
own trajectory shows a good model is being thrown away by late-epoch degradation. It is not
a legitimate *scheduling* move. If you shorten it, say why, and say in the report that
step-budget comparability is broken.

### Surviving a multi-hour run

Long runs are a discipline problem, not a science problem. The three things that make a
12-hour run survivable are all in [§1](#1-your-role-supervisor-not-operator):

- **tmux** ([§1.1](#11-process-isolation--read-first)) — the run outlives your session,
  your context compaction, and any tool timeout. This is the whole reason for the rule.
- **A `persistent: true` Monitor on phase changes** ([§8](#8-monitoring-correctly)) — silence
  for hours, one line when you are needed. No re-arming.
- **`/tmp/ap_supervisor_notes.md`** (rule 10) — two lines per round. If your context is
  compacted mid-run, this file *is* the run's memory. Write it every round, not at the end.

Also: verify each apply's diff **before** the training run consumes hours of GPU time
([hazard 1](#hazard-1-a-silent-no-op-apply)). A no-op round is cheap on CMNIST and very
expensive on a BREEDS case.

### One case per VM

The server holds a single `asyncio.Lock` across all long-running stages, so one case per API
server is the natural unit and needs no coordination. Per-VM checklist:

- Each VM needs the dataset for its case under `SUBPOP_DATA_DIR` (default
  `/mnt/workspace/data`); the shared-root assumption in `CASE.md` only matters when several
  cases share a box.
- `pipeline/stages.py::_train_interpreter()` resolves `<repo>/venv/bin/python` from the repo
  root. Override with `AUTOPROBE_TRAIN_PYTHON` if a case needs a different stack.
- Confirm the GPU is actually visible on that VM (`make doctor`) rather than assuming — see
  [hazard 4](#hazard-4-agents-assert-this-gpu-less-box--it-is-false).
- Batch size comes from the registry and is tuned for the benchmark, not for your GPU's
  memory. Raising it to fill an H100 changes the optimisation and breaks comparability;
  leave it alone.

---

## 3. Start the servers

Both go in tmux (rule 2). Never from a bare Bash call.

```bash
REPO=/mnt/workspace/AutoProbe
tmux new-session -d -s ap_api -c $REPO 'source venv/bin/activate && make api 2>&1 | tee /tmp/ap_api.log'
tmux new-session -d -s ap_web -c $REPO 'make web 2>&1 | tee /tmp/ap_web.log'
```

Then verify — readiness **and** isolation:

```bash
curl -s http://127.0.0.1:8765/api/health                      # {"ok":true,"busy":false}
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/
ps -o pid,ppid,cmd -p $(pgrep -f uvicorn | head -1)           # parent must be tmux, PPID 1
```

Report the isolation result. If the parent chain leads to a `claude` process, **stop and say
so** — the servers will die with your session.

`make api` auto-starts the ccr gateway. Fetch `openapi.json` once (rule 11), then drive
`:8765` only. `:3000` is the human's window.

---

## 4. Stage 1 — context and probe

```
POST /api/workspace/open          {"path": ".../cases/<Case>"}
POST /api/runs                    {"workspace": ".../cases/<Case>"}   -> run_id
POST /api/runs/{id}/stage1/context {"context": "..."}
POST /api/runs/{id}/stage1/generate
```

Generation blocks for ~5–10 min, so fire it through tmux (rule 4) and poll.

### Writing the context (this is the highest-leverage thing you type)

Read `CASE.md` and `train.py` first, then write ~3 500–4 000 characters covering:

1. **What `train.py` actually does** — dataset construction, how the spurious attribute is
   generated and how strongly it correlates with the label, split sizes, backbone,
   algorithm (`ERM` is the deliberate un-mitigated baseline), optimiser/hparams, step
   budget, and what gets written per epoch.
2. **`TRAIN_ATTR`.** If `"no"`, say explicitly that group attributes are *hidden during
   training* (the `a` column is zeroed for the train split only, val/test keep true
   attributes), so any fix must work without true group labels at train time, and the
   comparable published column is the attribute-unknown one.
3. **The problem being studied**, named: subpopulation shift; the model looks fine on
   average while a minority subpopulation collapses. Define the groups as the `(y, a)`
   cross product and give the majority/minority proportions.
4. **What you want**: a probe that *detects and quantifies* the failure, and then to
   actually *fix* it. State that `min_group.accuracy` is the headline,
   `adjusted_accuracy` is a lower-variance stand-in, and that `overall.accuracy` is only a
   utility floor — a probe tracking average accuracy alone would miss the failure entirely.
   Note that grouping by class alone is strictly weaker than the `(y,a)` cross product.

**Generation can fail with a JSON parse error** (the model returns prose). That is
transient — just POST `generate` again. It is not a crash and does not count against
rule 16's retry budget.

### Choosing the probe

You will get 10 candidates with confidences. **Pick the one whose tracked scalar is the
thing that must move**, not the highest confidence.

- ✅ A probe tracking **worst-group accuracy = `min_group.accuracy` over `(y,a)`**. That is
  the "model performs badly on a subpopulation" quantity, it is the benchmark's headline,
  and improving it *is* the fix.
- ❌ Diagnostics that merely *detect* the shortcut (colour-flip perturbation tests,
  feature-space separability, calibration audits). They often score higher confidence, but
  improving them does not raise minority performance, and the fix-loop optimises the
  tracked scalar.
- ❌ Anything grouping by class only, or tracking overall accuracy.

Say why you picked it in one sentence (rule 13).

---

## 5. Stage 2 — dev plan

```
POST /api/runs/{id}/stage2/generate
GET  /api/runs/{id}/stage2/artifact
POST /api/runs/{id}/stage2/select   {"index": N}
```

Three plans, each with `metric`, `standard_threshold` (PASS), `acceptable_threshold`.
Prefer the plan that **reads the benchmark's own field directly**
(`val["min_group"]["accuracy"]`) over one that hand-rolls a derived statistic (bootstrapped
lower bounds, Wilson intervals, etc.). Derived statistics drift from the benchmark
definition and make the later independent comparison harder.

**Understand what the thresholds are before you accept them.** Unless `CASE.md`'s
"Published reference numbers" table has been filled in, these numbers are *invented by the
LLM from a guess*, not taken from the paper. That is acceptable for a blind run — say so
plainly to the user rather than implying they are grounded.

Also sanity-check the ceiling. If the case injects label noise (CMNIST's
`cmnist_flip_prob = 0.25`), Bayes-optimal accuracy on any group is `1 - flip_prob` = 0.75,
so a 0.68 bar is 91 % of the ceiling — demanding. Knowing the ceiling stops you from
chasing an impossible target for hours.

---

## 6. Stage 3 — implement and baseline

```
POST /api/runs/{id}/stage3/implement
```

Agent writes `prober.py`, integrates `train.py`, then the orchestrator runs training once.
Arm the persistent monitor and wait (~1 training run of wall-clock).

**Verify the integration** (read-only) once phase reaches `post-impl-test-run`:

```bash
grep -n "from prober import\|record(\|conclude(" cases/<Case>/train.py
wc -l cases/<Case>/train.py            # ~ baseline + ~12 lines, not rewritten
grep -c "^class " cases/<Case>/train.py # inlined algorithm classes still present
```

When it finishes you get iteration row 1 — the baseline. **If it PASSes here**, stop and
tell the user: the threshold was too loose to be interesting, and there is nothing to
improve. That is a result about the threshold, not about the pipeline.

---

## 7. Stage 4 — drive it manually, not on auto-pilot

### Use the manual fix-plan flow

`POST /stage4/auto-fix-loop` is the "Start auto probe-fixing" button, but
`_at_terminal_state` stops at **4 iteration rows**, i.e. only ~3 fix attempts, and it
auto-picks by confidence. One wasted round (see
[hazard 1](#hazard-1-a-silent-no-op-apply)) leaves you with two. Instead drive the same path
the "Continue manually" modal uses:

```
POST /api/runs/{id}/stage4/fix-plans/generate   {"hint": "..."}
GET  /api/runs/{id}/stage4/fix-plans/artifact
POST /api/runs/{id}/stage4/fix-plans/select     {"index": N}
```

This is not a backend change — it is the documented manual flow, and you control the round
count. Budget 5–6 rounds. Both POSTs can block; use tmux (rule 4).

### The `hint` parameter is your main instrument

`fix-plans/generate` takes `{"hint": "..."}`, passed to the generator as non-binding user
direction. It is the frontend's hint box. Long, specific, evidence-backed hints work; vague
ones waste rounds. Structure that worked:

1. **Workspace note.** `train.py` is self-contained — every algorithm, network, optimiser
   and hparam default is inlined *in it*; edit them there. Dataset + `eval_helper` are
   imported and **frozen**. Never edit `prober.py` or the thresholds.
2. **What the probe has already measured** — paste the actual per-epoch series. This is
   the single most useful thing in the hint.
3. **The structural fact.** The decision statistic is `tail_mean` = mean of the **last 5
   epochs**. If the metric has a within-run trend, the run is graded on its most
   over-trained window. Spell this out.
4. **Ranked suggestions**, most principled first, each with its mechanism.
5. **Explicit dead ends**, with the measured numbers that killed them. This is what stops
   the agent burning a round re-testing something.
6. **Constraints**: keep `TRAIN_ATTR`; don't touch scorer/dataset/prober/thresholds; keep
   the case's frozen constants.
7. **"Actually write your edit to disk and produce the change_log for this round."**

### Ordering the attempts

1. **Attack the shortcut mechanism directly.** For a colour-spurious case, a train-only
   augmentation that decorrelates colour from the label (e.g. randomly swapping the R/G
   channels inside `ERM.update`, gated on `self.training` so eval is untouched). On CMNIST
   this alone took worst-group `0.1821 → 0.6592` — 94 % of the total gain — and held
   ~0.70 for eleven consecutive epochs under plain ERM.
2. **Robust objective / regularisation.** `ALGORITHM = "GroupDRO"` with a tuned
   `groupdro_eta`, stronger weight decay, augmentation.
3. **Early end, as an explicit fallback.** `MAX_EPOCHS` is a documented constant and
   nothing in the prompts forbids changing it; `main()` already does
   `n_epochs = min(n_epochs, MAX_EPOCHS)`. Legitimate early stopping, and the same logic as
   the benchmark's own model-selection criteria — but note in the change log that it
   trades away step-budget comparability.

Do 1 and 2 first so that if it passes you can say *which* mechanism did the work. Reach for
3 when the trajectory shows a good model existed early and was thrown away.

### After **every** apply, verify (read-only) before the training run finishes

```bash
cd cases/<Case>
git --git-dir=.agent_probe/snapshot.git --work-tree=. diff HEAD --stat -- train.py   # non-empty!
git --git-dir=.agent_probe/snapshot.git --work-tree=. diff HEAD --stat -- prober.py  # must be EMPTY
ls .agent_probe/change_log_*.txt
```

An empty `train.py` diff plus a missing `change_log_N.txt` means the apply was a no-op.
These are `diff`/`ls` only — they read, they do not modify, so rule 15 is satisfied. Do this
**before** hours of GPU time go into a round that changed nothing.

---

## 8. Monitoring correctly

Watch **state changes**, never streams (rule 8). Point Monitor at a poller over
`stage.json`, run it with `persistent: true`, and let it stay silent through the training.

```bash
# one line ONLY when phase changes; persistent:true so it outlives the run
R=<run_id>; S=response/$R/stage.json; prev=""
while true; do
  cur=$(python3 -c "
import json
d=json.load(open('$S'))
it=d['iterations']; last=it[-1] if it else {}
print(f\"{d['stage']}|{d['phase']}|{d['current_action']}|rows={len(it)}|tm={last.get('tail_mean')}|{last.get('status')}\")" 2>/dev/null)
  [ -n "$cur" ] && [ "$cur" != "$prev" ] && { echo "$cur"; prev="$cur"; }
  sleep 60
done
```

Rules that matter here:

- **`persistent: true`.** A single training run can exceed 5 hours; a non-persistent monitor
  would expire many times over and force you to re-arm.
- **Never point Monitor at `agent.log`** (rule 6), and never at raw training output.
- **`grep --line-buffered`** anywhere you pipe through grep.
- **Cover failure states in the exit condition**, not just success — silence must not look
  identical to "still running".
- **Per-epoch numbers are not events.** `probe_live.json` is for when a round *ends* or you
  need a number to decide (rule 9). Do not emit one line per epoch.
- On an *unexplained* phase change (`running` → `ready`/`input`, no new iteration row), make
  one `GET /api/runs/{id}` for `last_error` — see the caveat under rule 7.
- `TaskStop` your own stale monitors; never anything the pipeline owns (rule 14).

Training output is captured with `capture_output=True` and only written to `agent.log`
**after** the run finishes, so there is nothing to stream live anyway.

---

## 9. Known hazards (learned the hard way)

### Hazard 1: a silent no-op apply
An apply agent planned its edit correctly, then its completion was **truncated
mid-sentence** before it emitted the `Edit` call — and the CLI still exited
`subtype=success`. Tells: the re-run reproducing the previous round's per-epoch values
*bit-for-bit*, an empty `train.py` diff, and no `change_log_N.txt`. Nothing in the pipeline
detects this. **Always verify the diff** ([§7](#after-every-apply-verify-read-only-before-the-training-run-finishes))
— that check is cheap and does not require opening `agent.log`. Root-causing it *did*
require the log, but detection does not; if the user asks why, read the last 50 lines once
(rule 6). Related: the same route produced visibly corrupted output elsewhere
(`/和mnt/...`, `"titled"`, `agorit hs.py`).

### Hazard 2: `pkill -f "claude --output-format"` kills *you*
The pipeline's `claude` subprocesses and your own agent session share a binary path. That
pattern terminated the session mid-task. This is part of why rule 14 forbids killing
processes at all — pipeline subprocesses exit on their own.

### Hazard 3: revert only covers `train.py`
`.agent_probe/snapshot.git` tracks **exactly one file**, and
`_maybe_revert_on_regression` calls `snap.restore_train`. Any edit landing outside
`train.py` **survives its own revert** — the orchestrator logs "reverted" while the change
stays live, and the workspace silently disagrees with its recorded history. This is why the
algorithms are now inlined into `train.py`. If a plan proposes editing anything under
`subpopbench/`, redirect it in the hint.

Corollary: **do not resume an old run whose archived `snapshot.git` predates the inlining.**
Its `HEAD` holds the old wired `train.py`, so the first regression would overwrite the
self-contained file and re-introduce this bug. Start a fresh run instead.

### Hazard 4: agents assert "this GPU-less box" — it is false
`torch.cuda.is_available()` is `True` on these VMs. `CLAUDE.md`, the `Makefile` and the
`README` used to say "this box has no GPU"; that has been corrected, and `make setup` now
defaults to the `cu128` wheel index. Fix-plan agents still sometimes assert "this GPU-less
box" in their rationale — the reasoning is wrong even when the edit is harmless. Check
`make doctor` (torch must show a `+cuXXX` suffix) rather than trusting prose, and correct
the assumption in your next hint.

### Hazard 5: `tail_mean` grades the most over-trained window
The decision value is the mean of the **last 5 epochs**. When the metric peaks early and
decays, a genuinely good model is thrown away. On CMNIST, GroupDRO reached 0.69–0.73 by
epochs 2–5 and was scored 0.5084 on epochs 15–19. Read the whole per-epoch series at
round end — the *shape* tells you whether you are fighting overfitting, variance, or a
genuinely weak method.

### Hazard 6: distinguish a real trade-off from plain overfitting
Check whether overall accuracy actually *improves* while worst-group falls. On CMNIST it
did not — overall was flat (+0.003) while worst-group fell 0.294 and train loss dropped
45 %. That is memorisation, not a trade-off, and it calls for regularisation/early
stopping rather than a different robust objective.

### Hazard 7: the anchor guard
A round auto-reverts if `train.py`'s own metric (e.g. `val_loss`) degrades > 20 %, even if
the probe metric improved. Expect it; treat a breach as real evidence the change is bad.

---

## 10. Recovery

**While a run is live, you are read-only** (rule 15). Almost everything below is a *post-run*
action. When you do take one, **tell the user what you changed** — that is the condition on
which they are permitted.

### During a run — supervisor actions only

| Situation | Action |
|---|---|
| HTTP 409 | Normal: another stage holds the lock. Back off and re-poll (rule 12). |
| Stage-1/2 JSON parse error | POST `generate` again. Transient, not a crash. |
| `train.py` crashed | Do nothing. `PROMPT_EIGHT` retries up to 5×. Report after the fifth (rule 16). |
| Round regressed | Nothing — `_maybe_revert_on_regression` already restored `train.py`. Verify with a read-only `diff`. |
| Run wedged / should be stopped | **Say so and wait for the human.** Never `/api/cancel`, never kill (rule 14). |
| Unexplained phase change | One `GET /api/runs/{id}` for `last_error`. If still unclear, last 50 lines of `agent.log`, once. |
| Edit landed outside `train.py` | It escaped revert. Do **not** fix it mid-run — say so, and put the correction in the next hint. |

### After a run — repair, with notification

| Situation | Action |
|---|---|
| Need to clean but keep evidence | **Archive first**: `cp -a .agent_probe prober.py output response/<run_id>/workspace_artifacts_backup/`. `probe_result_N.json` and `change_log_N.txt` exist **only** in the workspace and are destroyed by cleanup. |
| Stale edit outside `train.py` | Restore from the snapshot git or from `cases_copy/<Case>`. |
| Case badly mangled | `git checkout -- cases/<Case>` (tracked files), then remove `prober.py`, `.agent_probe/`, `output/`, `__pycache__`. |
| Case unrecoverable | Restore from `cases_copy/<Case>` — pristine template, read-only source, never edit it. |

Run metadata under `response/<run_id>/` (`stage.json`, `fix_plans_N.json`, and `agent.log`
itself) survives workspace cleanup and is your audit trail.

---

## 11. Reporting and independent validation

Report each round as: change → `tail_mean` → status → kept/reverted. State plainly whether
it passed; do not dress a FAIL up as progress. Mirror the same two lines into
`/tmp/ap_supervisor_notes.md` (rule 10).

The independent check uses the **frozen** scorer's own output, already on disk:

```bash
cat cases/<Case>/output/final_results.json   # eval_metrics for va and te
cat cases/<Case>/output/history.json         # per-epoch
```

`final_results.json` carries the full benchmark metric set — `overall`, `per_group`,
`per_class`, `per_attribute`, `min_group`, `max_group`, `min_attr`, `adjusted_accuracy`.
Report the **held-out `te` split**.

**No model checkpoint is saved** (`train.py` drops checkpoint plumbing), so the weights
cannot be re-evaluated later without retraining. `final_results.json` is overwritten every
run, so it reflects only the most recent round — archive it if you need a specific round's
numbers.

Things worth calling out when you report:

- The **per-group table**. On the passing CMNIST run the worst group became the *largest*
  group (n=8009 @ 0.6913) while the colour-conflicting minorities scored 0.7053/0.7339 —
  the spurious structure had inverted out of existence, spread 0.0695. That is far more
  convincing than the headline scalar alone.
- **Val/test agreement** (0.6871 vs 0.6913) — shows the result is not a val-selection artefact.
- **Distance to ceiling** — 0.6913 is 92 % of the 0.75 label-noise ceiling.
- **Any comparability caveat**, unprompted. `MAX_EPOCHS=8` means ~2 216 steps vs the
  benchmark's 5 001; if a strictly comparable number is wanted, quote the full-budget round
  instead.

---

## 12. Worked example — CMNIST, the passing run

| Round | Change | `tail_mean` | Status |
|---|---|---|---|
| 1 | ERM baseline | 0.1821 | FAIL |
| 2 | R/G channel swap in `ERM.update`, p=0.5 | 0.6592 | FAIL, acceptable met — kept |
| 3 | same swap made per-sample | 0.6418 | FAIL — reverted |
| 4 | + `MAX_EPOCHS = 8` | **0.7080** | **PASS** |

Held-out test: worst-group **0.6913**, overall 0.7247, adjusted 0.7228.

Notes worth carrying to other cases. Round 2 did 94 % of the work and was the *principled*
fix. Round 3 is a useful negative: per-sample swapping raised epochs 1–11 but not the tail,
proving the late instability was not augmentation noise. Round 4 only harvested a model the
augmentation had already produced — so this is **not** a "passed by training less" result,
and you should be able to say which mechanism did the work.

Reuse across runs is fine and makes runs comparable: copy `probe_designs.json`,
`probe_confidenced.json`, `dev_doc.json`, `dev_doc_confidenced.json` from the old
`response/<run_id>/` into the new one, then call `stage1/select` and `stage2/select`
(they only read the confidenced artifact). Stage 3 must be re-run, because `prober.py` has
to be written against the current `train.py`. This is `response/` bookkeeping, not a
workspace edit
