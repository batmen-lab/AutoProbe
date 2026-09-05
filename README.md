[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21349577-blue)](https://doi.org/10.5281/zenodo.21349577)

# AutoProbe

An agentic pipeline that designs, implements, and iteratively improves
evaluation **probes** for ML training pipelines.

You point AutoProbe at a project folder containing a `train.py`. It then:

1. designs candidate probes (quantitative checks grounded in published methods),
2. picks one and turns it into runnable `prober.py` code,
3. integrates it into your `train.py`, and
4. iteratively rewrites `train.py` until the probe metric crosses a threshold —
   or you stop it.

**Day-to-day, you run two things in two terminals:**

```bash
make api      # terminal 1 — FastAPI on :8765
make web      # terminal 2 — Next.js on :3000  → open http://localhost:3000
```

That's it. Everything happens in the browser.

(There's also a `python main.py` CLI driver and a `python test.py` smoke
check — both wrap the same pipeline. They're optional. See [Other entry
points](#other-entry-points) at the bottom if you ever need them.)

---

## Demo videos

Two end-to-end screen recordings of the web UI live in [`utilities/video_demo/`](utilities/video_demo/):

- **[`utilities/video_demo/mimic_run.mp4`](utilities/video_demo/mimic_run.mp4)** — full normal-mode
  pipeline run on the MIMIC workspace: Probe Design → Dev Plan → Implementation
  → Probe Fixing, including the PASS / FAIL dialogs at the end of a round.
- **[`utilities/video_demo/autoresearch.mp4`](utilities/video_demo/autoresearch.mp4)** —
  auto-research mode: toggle on at Stage 1, agent writes `prober.py` and
  seeds the 10 `# potential_improvement_*` markers, then a batch of rounds
  runs with the orchestrator-level revert-on-regression and the monotonic
  per-run chart updating dot by dot.

Watch one or both before running the pipeline yourself — they cover the parts
of the UI that are easier to see than to describe (status bar phases, the
side-by-side live + per-run charts, the dialog flows).

---

## How the pipeline works

The system chains two kinds of `claude` subprocess calls:

- **NLP calls** — `claude -p --tools "" --no-session-persistence`. Short,
  JSON-returning, no filesystem access. Used for probe design, confidence
  scoring, plan generation.
- **Agent calls** — `claude -p --dangerously-skip-permissions`. Full tools, free
  to read and edit files inside the workspace. Used for code generation,
  integration, and iterative improvement.

Every call is a fresh subprocess: no shared session, no carried-over context.
Each one is streamed (`--output-format stream-json --verbose`) into a per-run
log file that the web UI tails over SSE.

```
User opens a workspace (folder w/ train.py)
        │
        ▼
Probe Design (Stage 1)
        ├─ Normal path:
        │     [NLP]  generate 10 probe designs     ← PROMPT_ONE
        │     [NLP]  score probe confidence (0–1)  ← PROMPT_TWO
        │     User picks 1 of 10
        │
        └─ Auto-research toggle → jump straight to Probe Fixing (see below)
        │
        ▼
Dev Plan (Stage 2)
         [NLP]  generate 3 dev plans               ← PROMPT_THREE
         [NLP]  score plan practicality            ← PROMPT_FOUR
         User picks 1 of 3 (optional: edit threshold)
        │
        ▼
Implementation (Stage 3)
         [Agent] write prober.py + integrate       ← PROMPT_FIVE
         [Run] python train.py
                └─ on crash → [Agent] fix          ← PROMPT_EIGHT  (≤5 retries)
        │
        ▼
Probe Fixing (Stage 4) — normal mode
         User clicks "Start auto probe-fixing":
           snapshot train.py
           [Agent] improve train.py                ← PROMPT_SEVEN
           [Run] python train.py
                └─ on crash → [Agent] fix          ← PROMPT_EIGHT
           Read .agent_probe/metric/probe_result_N.json → PASS / FAIL?
              ├─ PASS  → dialog: Discard & re-pick / Keep & re-baseline / Stay
              └─ FAIL  → dialog: Give up / Relax threshold / Continue probe-fixing
```

### Auto-research mode (Stage 1 alternative)

Toggling **auto-research** at Stage 1 collapses the first three stages into one
agent-driven setup, then parks the run at Stage 4 with a different UI:

```
User toggles "auto-research" at Probe Design
        │
        ▼
Auto-research Setup (single Stage-1 action)
         [Agent] pick a standard metric, write prober.py,    ← PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION
                 integrate it into train.py
         [Run]   validate the integration
         [Agent] seed 10 # potential_improvement_N: markers  ← PROMPT_SIX
                 into train.py
         [Run]   re-validate after the markers are in place
         Orchestrator cleans up so indexing aligns (iter K
         → train_version_K / probe_result_K / change_log_K).
        │
        ▼
Probe Fixing (Stage 4) — auto-research mode
         Sidebar locks Stage 2 and Stage 3 (skipped).
         User sets a "rounds" count (default 10) and clicks
         "Start auto-research (N rounds)". For each round:
           snapshot train.py
           [Agent] apply one targeted change                 ← PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT
                   (epoch budget enforced; size adapts to
                    recent progress; smooth increments)
           [Run]   python train.py
                  └─ on crash → [Agent] fix                  ← PROMPT_EIGHT
           Orchestrator compares new tail_mean to running
           best (respecting direction):
              ├─ improved → keep change, advance best
              └─ regressed → restore train.py from snapshot
                              (per-run chart stays monotonic)
        │
        ▼
Post-batch dialog: Run more rounds / Back to Probe Design (clean everything) / Stay
```

Key differences from normal mode: no PASS/FAIL threshold, no dev-plan
selection, the orchestrator is the source of truth for revert-on-regression
(not the agent), and the UI shows a second chart — one green dot per round —
that is monotonic by construction.

### Artifacts produced inside the workspace

| Path | Written by | Contents |
|---|---|---|
| `prober.py` | Stage 3 agent (or auto-research setup) | Probe definition; never touched again. |
| `train.py` | Stage 3 + 4 agents | Modified in place. The orchestrator may rewind it after a round (auto-research revert-on-regression, or the user's back-step). |
| `.agent_probe/snapshot/train_version_N.py` | pipeline | Snapshot of `train.py` before each agent edit. Used for revert and for auto-research's keep-best logic. After auto-research setup, indices align: round K writes `train_version_K` alongside `probe_result_K` and `change_log_K`. |
| `.agent_probe/metric/probe_result_N.json` | `prober.py` | metric_name, direction, per-epoch values, min/max/mean/std, `tail_mean`, threshold + status (normal mode only). |
| `.agent_probe/plot/probe_result_N.pdf` | `prober.py` (normal mode) | Plotly chart of the metric over epochs. Auto-research skips plots. |
| `.agent_probe/live/probe_live.json` | `prober.py` | Per-epoch trajectory updated during the run; powers the web UI's live chart. |
| `.agent_probe/change_log_N.txt` | Stage 4 agent | What changed each round; the next round reads these to avoid repeating an unsuccessful change. |

### Run state outside the workspace

Each run also has metadata under `response/<YYYYMMDDHHMMSS>/` at the repo root:

| File | Contents |
|---|---|
| `stage.json` | Current stage, phase, selections, iteration history. |
| `agent.log` | Full streamed transcript (all NLP + agent calls). |
| `probe_designs.json`, `probe_confidenced.json` | Stage 1 outputs. |
| `dev_doc.json`, `dev_doc_confidenced.json` | Stage 2 outputs. |

The pipeline also keeps a `response/_app_state.json` with the currently-open
workspace and recent-workspace history (VS Code-style).

`response/` is gitignored — runs are local.

---

## Setup

You need three things on the box:

1. **Python 3.12.** Not 3.13+: the SubpopBench case stack (`torch`,
   `torchvision`, `timm`, `netcal`) has no wheels for it yet. If the system
   python is newer, `make setup` uses [`uv`](https://docs.astral.sh/uv/) to
   fetch a 3.12 for the venv — nothing is installed system-wide.
2. **Node.js 18+**
3. The **`claude` CLI** (`@anthropic-ai/claude-code`), authenticated — or
   [claude-code-router](#model-routing-claude-code-router) in front of it.

### 1. Clone

```bash
git clone <your-fork-of-AutoProbe>.git
cd AutoProbe
```

### 2. Python environment

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python venv/bin/python -r requirements.txt
```

torch and torchvision come from the **CUDA 12.8** wheel index, which `make
setup` uses by default — these boxes have GPUs (an NVIDIA L4 on the reference
box), and the CPU build installs without error while training 10-50x slower.
On a genuinely GPU-less box, swap the index (`.../whl/cpu`) or let `make setup`
do it: `make setup PYTORCH_INDEX=https://download.pytorch.org/whl/cpu`.
Verify either way with `venv/bin/python -c "import torch;
print(torch.__version__, torch.cuda.is_available())"`.

`requirements.txt` covers the API server (`fastapi`, `uvicorn`, `pydantic`),
the training stack the agent leans on inside your project workspaces
(`numpy`, `pandas`, `scikit-learn`, `scipy`, `tqdm`, `transformers`, `plotly`,
`kaleido`, `matplotlib`) and the SubpopBench case stack (`timm`, `netcal`,
`pillow`).

> If your project workspace needs additional packages, install them in the
> **same** venv. `pipeline/stages.py` runs `train.py` with `venv/bin/python`
> explicitly — it resolves that path from the repo root rather than sniffing
> `PATH`, so it holds however you launched the server. Override it per-case
> with `AUTOPROBE_TRAIN_PYTHON=/path/to/other/venv/bin/python` when a
> workspace needs a stack that conflicts with the repo venv.

### 3. Node + web dependencies

```bash
node --version             # 18+
npm --version

cd web && npm install && cd ..
```

### 4. Install and authenticate the `claude` CLI

```bash
npm install -g @anthropic-ai/claude-code
claude --version           # verify
```

Authenticate one of two ways:

**Option A — API key.** Get one from <https://console.anthropic.com/>:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

**Option B — interactive OAuth (Claude Pro/Max).** Run `claude` once and follow
the browser flow, then `Ctrl-C` to exit:

```bash
claude
```

Both NLP calls and agent calls use the same auth.

### 5. Model routing (claude-code-router)

<a id="model-routing-claude-code-router"></a>

Every NLP and agent call is a `claude` subprocess, and the pipeline makes a
*lot* of them. [claude-code-router](https://github.com/musistudio/claude-code-router)
("ccr") puts a local gateway in front of the CLI that speaks the Anthropic
Messages API and forwards to a cheaper provider — this repo is currently
pointed at DeepSeek via OpenRouter. It is **optional**: skip this section and
the CLI talks to Anthropic with the auth from step 4.

```bash
npm install -g @musistudio/claude-code-router
ccr start --no-open
ccr ui                     # providers, models, routes, gateway key
```

In the UI, add a provider (OpenRouter, an OpenAI-compatible endpoint, …) with
the models you want, then set the **default** and **background** routes.
`make ccr-up` from then on just ensures the gateway is serving.

**How the wiring works.** `pipeline/router.py` reads ccr's config
(`~/.claude-code-router/config.sqlite`) and injects `ANTHROPIC_BASE_URL`,
`ANTHROPIC_AUTH_TOKEN` and the model aliases into every `claude` subprocess
`pipeline/llm.py` spawns. `llm.py` asks for `--model opus`; the CLI resolves
that alias against `ANTHROPIC_DEFAULT_OPUS_MODEL` *before* the request leaves
the box, so the gateway only ever sees a model your provider actually lists.
That last part matters — the ccr gateway does **not** alias `claude-*` names
onto your route, it 400s with *"Model … is not configured for target
provider"*.

Anything you export yourself wins over what `router.py` reads from the DB, and
`AUTOPROBE_ROUTER=off` disables the injection entirely:

```bash
AUTOPROBE_ROUTER=off make api      # straight to Anthropic
```

Setup details, the config schema, a headless configuration recipe and a
troubleshooting table live in **[CCR_and_openRouter.md](CCR_and_openRouter.md)**.

### 6. (Optional) Smoke test

```bash
venv/bin/python test.py
```

It prints the resolved routing first, then runs the same three call shapes the
pipeline uses — through the **same environment** `pipeline/llm.py` gives its
subprocesses, so it tests the routed path rather than ambient Anthropic auth.

```
router: ccr http://127.0.0.1:3456 default=<your model> small=<your model>

── NLP model (Claude, no tools) ────────
  PASS — got: {'status': 'ok', 'model': 'nlp'}
── Agent (Claude, full tools) ──────────
  PASS — got: 'PONG'
── Web search (informational — routed) ──
  FAIL — ...
```

The first two must pass — a failure there means the env isn't ready, fix it
before running the pipeline. The third is **informational**: `WebSearch` is
Anthropic's server-side search and returns nothing through a third-party
provider, which is exactly why `pipeline/llm.py` leaves it out of `NLP_TOOLS`
(`WebFetch` plus the local `Grep`/`Glob`/`Read` tools do the grounding the
prompts need). Expect it to fail on a routed setup.

### One-command setup (Makefile)

If you're on Linux/macOS and have `make`:

```bash
make setup        # uv venv on python 3.12, CPU torch, requirements.txt, npm install
make doctor       # verify python / node / claude / ccr wiring in one shot
```

`make doctor` prints the interpreter that will run `train.py`, the resolved ccr
gateway and routed models, and the installed torch/transformers versions. Run
it first whenever something looks wrong.

---

## Running

In two terminals (both with the venv activated):

```bash
# terminal 1
make api
# Uvicorn running on http://127.0.0.1:8765

# terminal 2
make web
# ▲ Next.js  →  http://localhost:3000
```

Open <http://localhost:3000> and:

1. **Open a project folder.** Click *Browse…* and pick a directory that
   contains a `train.py`. The most-recently-used folder is remembered.
2. **Create a new run** (or click an existing one in the resume list).
3. **Probe Design (Stage 1)** — type a 1–2 sentence description of the project +
   dataset, then either:
   - click *Generate Probes* and pick one of 10 candidates (normal path), **or**
   - flip the *auto-research mode* toggle and click *Run Auto-Research Setup*
     (the agent picks a metric and writes the prober for you; you'll land at
     Probe Fixing).
4. **Dev Plan (Stage 2)** — click *Generate Dev Plans*. Pick one of 3.
   Optionally edit the threshold before continuing. (Skipped in auto-research.)
5. **Implementation (Stage 3)** — click *Implement & Run*. The agent writes
   `prober.py`, integrates `train.py`, and runs it once. Watch the log dock at
   the bottom. (Skipped in auto-research.)
6. **Probe Fixing (Stage 4)** —
   - *Normal mode:* click *Start auto probe-fixing* to run one round, then
     respond to the PASS or FAIL dialog. PASS lets you discard or keep
     changes; FAIL offers Give up / Relax threshold / Continue.
   - *Auto-research mode:* set the number of rounds (default 10) and click
     *Start auto-research (N rounds)*. The orchestrator runs the batch with
     revert-on-regression. A post-batch dialog offers Run more / Back to
     Probe Design (clean everything) / Stay.

The persistent **status bar** above each stage tells you what the pipeline is
currently doing (e.g. *"Auto-research 3/10: applying one improvement to
train.py…"*) or what input it's waiting on (e.g. *"Waiting for probe
selection — pick one of the candidates below to continue."*).

The sidebar lets you go **back** to an earlier stage. The semantics are:
back to Probe Design or Dev Plan lands at the *end* of that stage (candidates
kept, your previous selection cleared so you can re-pick); back to
Implementation rebuilds prober/metrics. After a 4 → 1 back-step, the probe you
just tried is greyed out so you don't reselect it. The big red **Cancel**
button kills the active subprocess and resets the run's phase.

---

## Workspace requirements

A "workspace" = any directory containing a `train.py` that:

- runs an entire training loop when invoked as `python train.py` (no args),
- exits 0 on success, non-zero with a traceback on failure,
- imports whatever it needs — install those packages in the same venv.

Everything else (data loaders, preprocessing, helpers) lives alongside
`train.py` in that folder. The agent reads existing files before editing.

The repo ships 16 ready-made workspaces under [`cases/`](cases/), one per
SubpopBench subpopulation-shift case (`MIMICNotes`, `Waterbirds`, `CelebA`,
`CivilCommentsFine`, …). Each is standalone — it carries its own deep copy of
`subpopbench/`, so an agent editing one case cannot affect another — and each
has a `CASE.md` describing the shift type, the metric the probe should track,
the data setup, and where the paper's thresholds come from. Datasets are *not*
duplicated per case; every case reads one shared root via `SUBPOP_DATA_DIR`
(default `/mnt/workspace/data`).

`legacy/` holds the pre-`cases/` Kaggle/MIMIC workspaces and the notes from
past runs, kept for reference only.

---

## Project structure

```
AutoProbe/
├── main.py                         # optional CLI driver (same pipeline as the web UI)
├── test.py                         # claude-CLI smoke test
├── Questions.py                    # user-facing prompt strings (CLI)
├── requirements.txt                # Python deps (API + training + case stack)
├── Makefile                        # setup / doctor / api / web / ccr-up (cli is optional)
├── venv/                           # python 3.12 venv (gitignored) — runs the API *and* train.py
│
├── pipeline/                       # The actual pipeline
│   ├── __init__.py
│   ├── stages.py                   # generate_probes, select_probe, implement, fix-plan flow, …
│   ├── state.py                    # RunState — stage.json, snapshots, revert
│   ├── snapshot_git.py             # git-backed snapshots of the workspace
│   ├── workspace.py                # open / list workspaces (VS Code-style recent list)
│   ├── router.py                   # ccr v3 wiring — reads config.sqlite, builds the subprocess env
│   ├── llm.py                      # nlp_call / agent_call (subprocess + stream-json + cancel)
│   └── llm_codex.py                # same two entry points against the `codex` CLI (LLM_BACKEND=codex)
│
├── server/                         # FastAPI front-of-pipeline
│   └── app.py                      # /api/runs/<id>/stageN/..., SSE log, live metric, cancel
│
├── web/                            # Next.js 15 + React 19 + Tailwind
│   ├── package.json
│   └── src/
│       ├── app/                    # layout, globals, page.tsx
│       └── components/
│           ├── Home.tsx            # workspace + resume picker
│           ├── Sidebar.tsx         # stage navigator
│           ├── Stage1.tsx … Stage4.tsx
│           ├── LogPanel.tsx        # SSE log dock
│           ├── MetricChart.tsx     # live + completed metric chart
│           ├── WorkspaceBar.tsx    # folder browser
│           └── ui.tsx              # buttons, cards, pills
│
├── hard_prompt/                    # All system prompts (one per agent)
│   ├── nlp_prober_gen.py           # PROMPT_ONE   — 10 probe designs
│   ├── nlp_prober_confi_comput.py  # PROMPT_TWO   — confidence (0–1)
│   ├── nlp_dev_doc_gen.py          # PROMPT_THREE — 3 dev plans
│   ├── nlp_dd_confi_comput.py      # PROMPT_FOUR  — practicality scoring
│   ├── agent_dd_implement.py       # PROMPT_FIVE  — implement + integrate
│   ├── agent_improve_commentor.py  # PROMPT_SIX   — annotate train.py with 10 markers
│   ├── agent_iterat_improver.py    # PROMPT_SEVEN — iterate (normal mode)
│   ├── agent_exception_catcher.py  # PROMPT_EIGHT — fix crashed train.py
│   ├── agent_fix_plan_gen.py       # PROMPT_NINE / PROMPT_TEN — candidate fix plans + scoring
│   ├── agent_fix_plan_apply.py     # apply the selected fix plan
│   └── auto_research_prompt_patch.py
│                                   # Auto-research prompts: setup (write prober + pick metric)
│                                   # + iteration (one targeted change per round, epoch budget,
│                                   # smooth-progress heuristics; orchestrator handles revert).
│
├── cases/                          # Workspaces — one folder per SubpopBench case
│   └── MIMICNotes/                 # CASE.md, train.py, requirements.txt, vendored subpopbench/
│       ├── CASE.md                 # shift type, metric guidance, data setup, thresholds
│       └── train.py                # `python train.py`, no args — the workspace contract
│
├── projectinfo/                    # One-paragraph project descriptions (Stage-1 input)
├── utilities/video_demo/           # End-to-end UI screen recordings
│   ├── mimic_run.mp4               # Normal-mode pipeline on the MIMIC workspace
│   └── autoresearch.mp4            # Auto-research mode batch run
├── legacy/                         # Archived pre-`cases/` workspaces and past run notes
│
└── response/                       # Per-run metadata (gitignored)
    ├── _app_state.json             # current + recent workspaces
    └── <YYYYMMDDHHMMSS>/
        ├── stage.json              # run state — single source of truth
        ├── agent.log               # streamed transcript of all calls
        ├── probe_designs.json
        ├── probe_confidenced.json
        ├── dev_doc.json
        └── dev_doc_confidenced.json
```

---

## Key design decisions

- **Stateless calls, stateful runs.** Every NLP/agent invocation is a brand-new
  subprocess with `--no-session-persistence`. Continuity across stages comes
  from JSON files in `response/<run_id>/`, not from the model's session.
- **Confidence is supervisor-scored.** PROMPT_TWO and PROMPT_FOUR are separate
  agents from the generators — they fill in confidence independently to avoid
  self-assessment bias.
- **Frozen probe.** Once `prober.py` is written in Stage 3, the iterator
  (PROMPT_SEVEN) is told never to touch it. Only `train.py` changes.
- **Snapshot before edit.** Every iteration saves the current `train.py` to
  `.agent_probe/snapshot/train_version_N.py` *before* the agent runs.
  `revert_to(stage)` restores from these snapshots.
- **Single active stage.** The FastAPI server has one `asyncio.Lock` covering
  all long-running stages. Concurrent calls return 409. The Cancel endpoint
  kills the in-flight subprocess and resets the owning run's phase.
- **Live metric.** `prober.py` is asked to write
  `.agent_probe/live/probe_live.json` after each epoch. The web UI polls it
  during a run and falls back to the latest `probe_result_N.json` after the
  run completes.
- **Crash-tolerant runs.** `train.py` failures trigger PROMPT_EIGHT (up to 5
  retries per stage). Partial probe artifacts from failed attempts are purged
  before each retry so charts don't reflect dead code.
- **Normal-mode PASS gate.** Probe Fixing exits as soon as `probe_result_N.json`
  reports `"status": "PASS"`, and the UI pops a PASS dialog with three exits
  (Discard & re-pick / Keep & re-baseline / Stay). On FAIL, a separate dialog
  offers Give up / Relax threshold / Continue.
- **Auto-research keeps only the best version.** In auto-research mode there
  is no PASS threshold. After every round the orchestrator compares the new
  `tail_mean` against the running best (respecting `direction`). If the round
  didn't improve, the orchestrator rewinds `train.py` to that round's
  pre-iteration snapshot. The per-run chart in the UI is monotonic by
  construction — the agent doesn't have to do its own revert math.
- **Locked stages in auto-research.** Dev Plan and Implementation are not used
  in auto-research; the sidebar greys them out so the user can't navigate
  there. The only "back" path from Probe Fixing is all the way to Probe
  Design (which wipes everything and starts over).
- **Smooth-progress prompt heuristics.** The auto-research iteration prompt
  has a strict epoch budget (no inflating training epochs as a hack) and
  modulates step size based on the most recent round's delta — small steps
  when progress is positive, larger steps when it stalled. The goal is
  traceable round-by-round increments so the per-run chart tells a coherent
  story.
- **Back-step semantics (normal mode).** Back to Probe Design or Dev Plan
  lands at the *end* of that stage with the previous selection cleared, so
  the user can re-pick from the same candidate list. A 4 → 1 back-step also
  greys out the just-tried probe in the candidate list.

---

## Other entry points

You won't normally need these. The web UI is the primary interface.

**`python main.py`** — interactive CLI driver for the same pipeline. Same
4 stages, terminal prompts instead of buttons. Useful for scripted/headless
runs.

```bash
python main.py                                       # interactive
python main.py --workspace ./mimic --iterations 3    # non-interactive
python main.py --resume <run_id>                     # resume
python main.py list                                  # list runs
python main.py revert <run_id> --to-stage 3          # rewind a run
```

**`python test.py`** — smoke-tests the `claude` CLI (NLP call, agent call, web
search). Run this once after setup to confirm auth works. All three checks
should print `PASS`.

---

## Troubleshooting

| Symptom | Probable cause |
|---|---|
| `claude: command not found` | Step 4 wasn't done. Re-run `npm install -g @anthropic-ai/claude-code`. |
| `test.py` agent test fails with auth error | `ANTHROPIC_API_KEY` not set, or OAuth wasn't completed. Run `claude` once. Routing through ccr? Run `make doctor` — if it says `router: UNAVAILABLE` the gateway key or default route is missing (`ccr ui`). |
| `Model "claude-…" is not configured for target provider` | A `claude-*` model name reached the ccr gateway. v3 does not alias those onto your route — check `make doctor` shows a routed model, and don't export `ANTHROPIC_MODEL` by hand. |
| `ccr status` prints `Profile "status" was not found` | v1 muscle memory. v3 has no `status`/`restart`/`activate`. Use `make ccr-up` (it probes `/health`) and `ccr ui`. |
| `ModuleNotFoundError` the moment `train.py` starts | `train.py` ran under the wrong interpreter. `make doctor` prints the one that will be used; install the missing package into that venv, or set `AUTOPROBE_TRAIN_PYTHON`. |
| API returns 409 on stage call | Another stage is already running. Wait, or hit `/api/cancel`. |
| `Workspace missing train.py` | The folder you opened doesn't have a `train.py`. |
| Live chart stays empty during Stage 3/4 | `prober.py` isn't writing `.agent_probe/live/probe_live.json` — the chart will fill in once the run completes from `probe_result_N.json`. |
| Training crashes in a loop | Up to 5 auto-fix retries, then it stops. Check `agent.log` for the agent's reasoning. |

---

## License

See [LICENSE](LICENSE).
