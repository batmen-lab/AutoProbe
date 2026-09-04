# Auto-running a SubpopBench case, end to end

Operational runbook for an agent asked to *"auto-run AutoProbe on `cases/<Case>` like a
human user."* It assumes you have already read [`CLAUDE.md`](CLAUDE.md) (which tells you
the stale Stage-4 CLI engine is off-limits) and [`README.md`](README.md).

This procedure was derived from two full CMNIST runs. The second one **passed**:
worst-group accuracy `0.1821 → 0.7080` (PASS bar `0.68`), held-out test worst-group
`0.6913`. The first one failed and the post-mortem is what most of the warnings below
are made of. Read [Known hazards](#8-known-hazards-learned-the-hard-way) before you
start — several of them cost hours.

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

## 1. Pre-flight

```bash
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
| `cases_copy/` | The recovery copy. Read it, restore *from* it, never edit it. |
| `subpopbench/dataset/datasets.py` | Defines the task, including the spurious correlation. Editing it changes the problem instead of solving it. |
| `subpopbench/utils/eval_helper.py` | `eval_metrics` is **the scorer**. Must stay definitionally identical to the benchmark and independently auditable. |
| `prober.py` (once Stage 3 writes it) | The frozen probe. Thresholds live inside it precisely so the fix-loop cannot move the goalposts. |
| `pipeline/`, `server/`, `web/` | Backend. Drive it, do not modify it. |
| `main.py` Stage-4 loop, `POST /stage4/iterate` | Stale engine. See `CLAUDE.md`. |

Everything an improvement round is *allowed* to change now lives in the case's
`train.py`, which is self-contained (~2050 lines: hparams, optimizers, networks,
joint-DRO, all 25 algorithms inlined). That is deliberate — see
[hazard 3](#hazard-3-revert-only-covers-trainpy).

### Data availability

Only `cmnist`, `metashift`, `waterbirds` are present under `/mnt/workspace/data` by
default. Others need:

```bash
export SUBPOP_DATA_DIR=/mnt/workspace/data
python -m subpopbench.scripts.download --data_path $SUBPOP_DATA_DIR --download
```

`MIMICNoFinding`, `CheXpertNoFinding`, `CXRMultisite`, `MIMICNotes` need **credentialed
manual download** and will not work without it.

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

The L4 column is a *reference point for one shared box*, not a constraint. The intended
deployment is **one VM per case**, on dedicated A100/H100-class hardware, so wall-clock is
not a reason to change the experiment.

**Therefore: keep the paper-comparable step budget.** Do not lower `TOTAL_STEPS` or set
`MAX_EPOCHS` for turnaround — the whole point of these runs is a number that can be
compared against the published tables, and a shortened budget forfeits that. Reducing the
budget is a legitimate *research* move (early stopping, [§6](#ordering-the-attempts)) when
the probe's own trajectory shows a good model is being thrown away by late-epoch
degradation. It is not a legitimate *scheduling* move. If you shorten it, say why, and say
in the report that step-budget comparability is broken.

What long runs *do* change is your operational discipline, not the science:

- Multiply per-run time by 4–6 for a full Stage-3 + Stage-4 sequence, and tell the user the
  estimate up front so they can plan the VM, not so they can approve a shortcut.
- Re-arm monitors more often; a 9-hour run outlives ~9 monitor windows
  ([§7](#7-monitoring-correctly)).
- Verify each apply's diff *before* committing hours of GPU time to it
  ([hazard 1](#hazard-1-a-silent-no-op-apply)). A no-op round is cheap on CMNIST and very
  expensive on a BREEDS case.

### One case per VM

The server holds a single `asyncio.Lock` across all long-running stages, so one case per
API server is the natural unit and needs no coordination. Per-VM checklist:

- Each VM needs the dataset for its case under `SUBPOP_DATA_DIR` (default
  `/mnt/workspace/data`); the shared-root assumption in `CASE.md` only matters when several
  cases share a box.
- `pipeline/stages.py::_train_interpreter()` resolves `<repo>/venv/bin/python` from the repo
  root. Override with `AUTOPROBE_TRAIN_PYTHON` if a case needs a different stack.
- Confirm the GPU is actually visible on that VM (`make doctor`) rather than assuming — see
  [hazard 4](#hazard-4-claudemd-says-no-gpu--it-is-stale).
- Batch size comes from the registry and is tuned for the benchmark, not for your GPU's
  memory. Raising it to fill an H100 changes the optimisation and breaks comparability;
  leave it alone.

---

## 2. Start the servers

```bash
cd /mnt/workspace/AutoProbe && nohup make api > /tmp/.../api.log 2>&1 &
( cd /mnt/workspace/AutoProbe && nohup make web > /tmp/.../web.log 2>&1 & )
```

**The subshell parentheses on `make web` matter.** `cd X && nohup make api ... &`
backgrounds the *whole* `cd && make`, so the `cd` never applies to your shell, and a
following bare `make web` runs in `$HOME` and dies with
`No rule to make target 'web'`.

Wait for readiness with an `until` loop (see [§7](#7-monitoring-correctly)), then:

```bash
curl -s http://127.0.0.1:8765/api/health          # {"ok":true,"busy":false}
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/
```

`make api` auto-starts the ccr gateway. Everything below is plain HTTP against
`:8765` — the exact routes the frontend buttons call.

---

## 3. Stage 1 — context and probe

```bash
curl -s -X POST .../api/workspace/open  -d '{"path":".../cases/<Case>"}'
curl -s -X POST .../api/runs            -d '{"workspace":".../cases/<Case>"}'   # -> run_id
curl -s -X POST .../api/runs/$R/stage1/context -d @context.json
curl -s -X POST .../api/runs/$R/stage1/generate -d '{}'
```

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

Generation is two NLP calls and takes ~5–10 min. **It can fail with a JSON parse error**
(the model returns prose). That is transient — just POST `generate` again.

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

Say why you picked it, in one line, for the record.

---

## 4. Stage 2 — dev plan

```bash
curl -s -X POST .../api/runs/$R/stage2/generate -d '{}'
curl -s     .../api/runs/$R/stage2/artifact
curl -s -X POST .../api/runs/$R/stage2/select -d '{"index":N}'
```

Three plans, each with `metric`, `standard_threshold` (PASS), `acceptable_threshold`.
Prefer the plan that **reads the benchmark's own field directly** (`val["min_group"]["accuracy"]`)
over one that hand-rolls a derived statistic (bootstrapped lower bounds, Wilson intervals,
etc.). Derived statistics drift from the benchmark definition and make the later
independent comparison harder.

**Understand what the thresholds are before you accept them.** Unless `CASE.md`'s
"Published reference numbers" table has been filled in, these numbers are *invented by the
LLM from a guess*, not taken from the paper. That is acceptable for a blind run — say so
plainly to the user rather than implying they are grounded.

Also sanity-check the ceiling. If the case injects label noise (CMNIST's
`cmnist_flip_prob = 0.25`), Bayes-optimal accuracy on any group is `1 - flip_prob` = 0.75,
so a 0.68 bar is 91 % of the ceiling — demanding. Knowing the ceiling stops you from
chasing an impossible target for hours.

---

## 5. Stage 3 — implement and baseline

```bash
curl -s -X POST .../api/runs/$R/stage3/implement -d '{}'
```

Agent writes `prober.py`, integrates `train.py`, then the orchestrator runs training once.
Arm a monitor and wait (~1 run of wall-clock).

**Verify the integration** once it moves to `post-impl-test-run`:

```bash
grep -n "from prober import\|record(\|conclude(" train.py
wc -l train.py                          # should be ~ baseline + ~12 lines, not rewritten
grep -c "^class " train.py              # inlined algorithm classes still present
```

When it finishes you get iteration row 1 — the baseline. **If it PASSes here**, stop and
tell the user: the threshold was too loose to be interesting, and there is nothing to
improve. That is a result about the threshold, not about the pipeline.

---

## 6. Stage 4 — drive it manually, not on auto-pilot

### Use the manual fix-plan flow

`POST /stage4/auto-fix-loop` is the "Start auto probe-fixing" button, but
`_at_terminal_state` stops at **4 iteration rows**, i.e. only ~3 fix attempts, and it
auto-picks by confidence. One wasted round (see [hazard 1](#hazard-1-a-silent-no-op-apply))
leaves you with two. Instead drive the same path the "Continue manually" modal uses:

```bash
# generate 3 candidate plans, with your hint
curl -s -X POST .../api/runs/$R/stage4/fix-plans/generate -d @hint.json
# read them
curl -s     .../api/runs/$R/stage4/fix-plans/artifact
# apply one
curl -s -X POST .../api/runs/$R/stage4/fix-plans/select -d '{"index":N}'
```

This is not a backend change — it is the documented manual flow, and you control the round
count. Budget 5–6 rounds.

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

### After **every** apply, before the training run finishes, verify

```bash
cd cases/<Case>
git --git-dir=.agent_probe/snapshot.git --work-tree=. diff HEAD -- train.py   # non-empty!
git --git-dir=.agent_probe/snapshot.git --work-tree=. diff HEAD -- prober.py  # must be EMPTY
ls .agent_probe/change_log_*.txt
```

An empty `train.py` diff plus a missing `change_log_N.txt` means the apply was a no-op.

---

## 7. Monitoring correctly

**Bare foreground `sleep` is blocked.** Use `Monitor` with a poll loop, or `Bash` with
`run_in_background: true` and an `until` loop.

- **One notification** ("tell me when the server is up") → background `until` loop that
  exits.
- **Per-epoch progress** → `Monitor` polling the run state and
  `.agent_probe/live/probe_live.json`.

Training output is captured with `capture_output=True` and only written to `agent.log`
**after** the run finishes, so tailing `agent.log` shows nothing live. Poll
`probe_live.json` (per-epoch, written by `prober.py`) instead.

A monitor template that works:

```bash
while true; do
  st=$(curl -s --max-time 10 $API/api/runs/$R | python3 -c "...phase|action|rows|tail_mean|status|err...")
  ep=$(python3 -c "...len(values), last value from probe_live.json...")
  line="$st|$ep"; [ "$line" != "$prev" ] && { echo "$line"; prev="$line"; }
  case "$st" in running*|fix-plans-ready*) : ;; *) echo "ENDED $st"; break ;; esac
  sleep 90
done
```

Cover failure states in the exit condition, not just success — silence must not look
identical to "still running".

**Monitors cap at 1 hour.** A 50-minute training run plus agent time will outlive one.
When it expires, re-check state and re-arm; use a longer `sleep` (90–150 s) so one window
covers more. Stop stale monitors with `TaskStop` — two live monitors duplicate every event.

---

## 8. Known hazards (learned the hard way)

### Hazard 1: a silent no-op apply
An apply agent planned its edit correctly, then its completion was **truncated
mid-sentence** before it emitted the `Edit` call — and the CLI still exited
`subtype=success`. Tells: `out=3494` tokens vs ~24 000 for other applies, text ending
mid-word, and the re-run reproducing the previous round's per-epoch values *bit-for-bit*.
Nothing in the pipeline detects this. **Always verify the diff.** Related: the same route
produced visibly corrupted output elsewhere (`/和mnt/...`, `"titled"`, `agorit hs.py`).

### Hazard 2: `pkill -f "claude --output-format"` kills *you*
The pipeline's `claude` subprocesses and your own agent session share a binary path. That
pattern terminated the session mid-task. Pipeline subprocesses exit on their own when you
cancel; if you must clean up, match on PID, not on that pattern.

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

### Hazard 4: `CLAUDE.md` says "no GPU" — it is stale
The box has an NVIDIA L4 and `torch.cuda.is_available()` is `True`. Fix-plan agents
sometimes assert "this GPU-less box" in their rationale; the reasoning is wrong even when
the edit is harmless. Check `make doctor` rather than trusting prose.

### Hazard 5: `tail_mean` grades the most over-trained window
The decision value is the mean of the **last 5 epochs**. When the metric peaks early and
decays, a genuinely good model is thrown away. On CMNIST, GroupDRO reached 0.69–0.73 by
epochs 2–5 and was scored 0.5084 on epochs 15–19. Always print the whole per-epoch series
before deciding what to try next — the *shape* tells you whether you are fighting
overfitting, variance, or a genuinely weak method.

### Hazard 6: distinguish a real trade-off from plain overfitting
Check whether overall accuracy actually *improves* while worst-group falls. On CMNIST it
did not — overall was flat (+0.003) while worst-group fell 0.294 and train loss dropped
45 %. That is memorisation, not a trade-off, and it calls for regularisation/early
stopping rather than a different robust objective.

### Hazard 7: the anchor guard
A round auto-reverts if `train.py`'s own metric (e.g. `val_loss`) degrades > 20 %, even if
the probe metric improved. Expect it; treat a breach as real evidence the change is bad.

---

## 9. Recovery

| Situation | Action |
|---|---|
| Action wedged / must stop | `POST /api/cancel` (the red Cancel button). Kills the subprocess, resets phase. |
| Stage-2 JSON parse error | Just POST `generate` again. Transient. |
| Round regressed | Nothing — `_maybe_revert_on_regression` already restored `train.py`. Verify it did. |
| Edit landed outside `train.py` | It escaped revert. Restore by hand from the snapshot git and say so in the next hint. |
| Case badly mangled | `git checkout -- cases/<Case>` (tracked files), then delete `prober.py`, `.agent_probe/`, `output/`, `__pycache__`. |
| Case unrecoverable | Restore from `cases_copy/<Case>` — read-only source, never edit it. |
| Need to clean but keep evidence | **Archive first**: `cp -a .agent_probe prober.py output response/<run_id>/workspace_artifacts_backup/`. `probe_result_N.json` and `change_log_N.txt` exist *only* in the workspace. |

Run metadata under `response/<run_id>/` (`stage.json`, `agent.log`, `fix_plans_N.json`)
survives workspace cleanup and is your audit trail.

---

## 10. Reporting and independent validation

Report each round as: change → `tail_mean` → status → kept/reverted. State plainly whether
it passed; do not dress a FAIL up as progress.

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
- **Val/test agreement** (0.6871 vs 0.6913) — shows the result is not val-selection artefact.
- **Distance to ceiling** — 0.6913 is 92 % of the 0.75 label-noise ceiling.
- **Any comparability caveat**, unprompted. `MAX_EPOCHS=8` means ~2 216 steps vs the
  benchmark's 5 001; if a strictly comparable number is wanted, quote the full-budget round
  instead.

---

## 11. Worked example — CMNIST, the passing run

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
to be written against the current `train.py`.

---

## 12. Scaling to the other cases

CMNIST is a synthetic sanity check. The paper's substantive results are on the real
datasets, and beating those is the valuable claim. Carrying this over:

- **Re-derive the context per case.** The shift type differs (SC vs attribute imbalance vs
  class imbalance); `CASE.md` names the dominant one. `MIMICNotes` and `CXRMultisite` use
  **worst-attribute AUROC** (`min_attr.AUROC`), not worst-group accuracy — the probe must
  track the right headline metric.
- **The shortcut-augmentation trick is CMNIST-specific.** Colour is a synthetic, exactly
  invertible attribute. On Waterbirds (background), CelebA (hair/gender), MultiNLI /
  CivilComments (text) there is no clean channel swap. Expect the ranked list to lead with
  robust objectives (GroupDRO, reweighting, DFR-style last-layer retraining) and
  regularisation instead.
- **Text cases** (`MultiNLI`, `CivilCommentsFine`) run the BERT path: `optimizer=adamw`,
  `batch_size=32`, `last_layer_dropout=0.5`, 30 001 steps. Do not assume the image path —
  constructing a featurizer with `data_type="images"` on these trips
  `assert hparams['last_layer_dropout'] == 0.` in `ResNet.__init__`. That assert is
  original benchmark behaviour, not a bug you introduced.
- **Check the ceiling and the minority group size per case.** Worst-group accuracy is a min
  over groups, so its variance is set by the *smallest* group. `CASE.md` has a "Noise band"
  section for exactly this; measure it from the baseline run before trusting a tight
  threshold.
- **Do not trade comparability for wall-clock.** A 60 001-step case is ~12× CMNIST per
  round, but with one VM per case that is a scheduling fact, not a reason to shrink the
  budget. Run it at the published step count and let it take as long as it takes.
