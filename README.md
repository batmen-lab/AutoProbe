# AutoProbe

An agentic pipeline that designs, implements, and iteratively improves
evaluation **probes** for ML training pipelines.

You point AutoProbe at a project folder containing a `train.py`. It then:

1. designs candidate probes (quantitative checks grounded in published methods),
2. picks one and turns it into runnable `prober.py` code,
3. integrates it into your `train.py`, and
4. iteratively rewrites `train.py` until the probe metric crosses a threshold —
   or you stop it.

It ships with three ways to drive the same pipeline:

| Interface | Process | Use it for |
|---|---|---|
| **Web UI** (Next.js + FastAPI) | `make api` + `make web` | Normal use. Live log dock, live metric chart, click-driven stage navigation, revert. |
| **CLI** (`main.py`) | `make cli` | Scripted/headless runs. |
| **Smoke test** (`test.py`) | `python test.py` | Verify the `claude` CLI is reachable before a real run. |

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
Stage 1  [NLP]  generate 10 probe designs       ← PROMPT_ONE
         [NLP]  score probe confidence (0–1)    ← PROMPT_TWO
         User picks 1 of 10
        │
        ▼
Stage 2  [NLP]  generate 3 dev plans            ← PROMPT_THREE
         [NLP]  score plan practicality         ← PROMPT_FOUR
         User picks 1 of 3 (optional: edit threshold)
        │
        ▼
Stage 3  [Agent] write prober.py + integrate    ← PROMPT_FIVE
         [Run] python train.py
                └─ on crash → [Agent] fix       ← PROMPT_EIGHT  (≤5 retries)
        │
        ▼
Stage 4  Loop N times (default 3, early-stops on PASS):
           snapshot train.py
           [Agent] improve train.py             ← PROMPT_SEVEN
           [Run] python train.py
                └─ on crash → [Agent] fix       ← PROMPT_EIGHT
           Read .agent_probe/metric/probe_result_N.json → PASS/FAIL?
```

There is also an **auto-research mode** (Stage 1 alternative) that skips probe
selection entirely: the agent picks a standard performance metric, writes
`prober.py`, runs once, drops 10 `# potential_improvement_N:` comments into
`train.py` via PROMPT_SIX, and parks the run at Stage 4.

### Artifacts produced inside the workspace

| Path | Written by | Contents |
|---|---|---|
| `prober.py` | Stage 3 agent | Probe definition; never touched again. |
| `train.py` | Stage 3 + 4 agents | Modified in place. |
| `.agent_probe/snapshot/train_version_N.py` | pipeline | Snapshot of `train.py` before each agent edit. Used for revert. |
| `.agent_probe/metric/probe_result_N.json` | `prober.py` | metric_name, per-epoch values, min/max/mean/std, threshold, status (PASS/FAIL). |
| `.agent_probe/plot/probe_result_N.pdf` | `prober.py` | Plotly chart of the metric over epochs. |
| `.agent_probe/live/probe_live.json` | `prober.py` | Per-epoch trajectory updated during the run; powers the web UI's live chart. |
| `.agent_probe/change_log_N.txt` | Stage 4 agent | What changed each iteration; next iteration reads this. |

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

1. **Python 3.10+**
2. **Node.js 18+**
3. The **`claude` CLI** (`@anthropic-ai/claude-code`), authenticated.

### 1. Clone

```bash
git clone <your-fork-of-AutoProbe>.git
cd AutoProbe
```

### 2. Python environment

```bash
python3 --version          # must be 3.10+

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` covers both the API server (`fastapi`, `uvicorn`,
`pydantic`) and the training stack the agent will lean on inside your project
workspaces (`torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `tqdm`,
`transformers`, `plotly`, `kaleido`).

> If your project workspace needs additional packages, install them in the
> **same** venv. The pipeline runs `python train.py` from the venv.

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

### 5. (Optional) Smoke test

```bash
python test.py
```

Expected output (all three should print `PASS`):

```
── NLP model (Claude, no tools) ────────
  PASS — got: {'status': 'ok', 'model': 'nlp'}
── Agent (Claude, full tools) ──────────
  PASS — got: 'PONG'
── Web search (NLP, CRWV stock price) ──
  PASS — CRWV price: 123.45 (source: ...)
```

Failures here mean the env isn't ready — fix this before running the pipeline.

### One-command setup (Makefile)

If you're on Linux/macOS and have `make`:

```bash
make setup        # creates venv, installs requirements.txt, runs npm install
```

Then later:

```bash
make api          # FastAPI on :8765
make web          # Next.js dev server on :3000
make cli          # interactive CLI
```

---

## Running

### Web UI (recommended)

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
3. **Stage 1** — type a 1–2 sentence description of the project + dataset.
   Click *Generate*. Pick one of 10 probes.
4. **Stage 2** — click *Generate*. Pick one of 3 dev plans. Optionally edit the
   threshold before stage 3.
5. **Stage 3** — click *Implement*. The agent writes `prober.py`, modifies
   `train.py`, and runs it once. Watch the log dock at the bottom.
6. **Stage 4** — click *Iterate* one or more times. Stops early on PASS.

The sidebar lets you **revert** to an earlier stage (wipes that stage's
outputs and everything after it; keeps stage *inputs* so you can edit and
re-run). The big red **Cancel** button kills the active subprocess and resets
the run's phase.

### CLI

Same pipeline, no browser:

```bash
make cli
# or:  python main.py
# or:  python main.py --workspace ./mimic --iterations 3
# or:  python main.py --resume 20260506174444
```

```bash
python main.py list                                  # list runs
python main.py revert <run_id> --to-stage 3          # rewind a run
```

---

## Workspace requirements

A "workspace" = any directory containing a `train.py` that:

- runs an entire training loop when invoked as `python train.py` (no args),
- exits 0 on success, non-zero with a traceback on failure,
- imports whatever it needs — install those packages in the same venv.

Everything else (data loaders, preprocessing, helpers) lives alongside
`train.py` in that folder. The agent reads existing files before editing.

The repo ships with empty placeholder folders for several Kaggle/MIMIC-style
projects (`mimic/`, `home_credit/`, `m5_forecast/`, etc.). They're empty in git
and will be populated locally when you put your project in them — the
`.gitignore` keeps generated files out.

---

## Project structure

```
AutoProbe/
├── main.py                         # CLI driver
├── test.py                         # claude-CLI smoke test
├── Questions.py                    # user-facing prompt strings (CLI)
├── requirements.txt                # Python deps (API + training stack)
├── Makefile                        # setup / api / web / cli
│
├── pipeline/                       # The actual pipeline
│   ├── __init__.py
│   ├── stages.py                   # generate_probes, select_probe, implement, iterate_once, …
│   ├── state.py                    # RunState — stage.json, snapshots, revert
│   ├── workspace.py                # open / list workspaces (VS Code-style recent list)
│   └── llm.py                      # nlp_call / agent_call (subprocess + stream-json + cancel)
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
│   ├── agent_improve_commentor.py  # PROMPT_SIX   — annotate train.py
│   ├── agent_iterat_improver.py    # PROMPT_SEVEN — iterate
│   ├── agent_exception_catcher.py  # PROMPT_EIGHT — fix crashed train.py
│   └── auto_research_prompt_patch.py
│
├── response/                       # Per-run metadata (gitignored)
│   ├── _app_state.json             # current + recent workspaces
│   └── <YYYYMMDDHHMMSS>/
│       ├── stage.json              # run state — single source of truth
│       ├── agent.log               # streamed transcript of all calls
│       ├── probe_designs.json
│       ├── probe_confidenced.json
│       ├── dev_doc.json
│       └── dev_doc_confidenced.json
│
└── mimic/  home_credit/  m5_forecast/  …   # empty placeholders for project workspaces
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
- **Early stop.** Stage 4 exits as soon as `probe_result_N.json` reports
  `"status": "PASS"`, regardless of the configured iteration count.

---

## Troubleshooting

| Symptom | Probable cause |
|---|---|
| `claude: command not found` | Step 4 wasn't done. Re-run `npm install -g @anthropic-ai/claude-code`. |
| `test.py` agent test fails with auth error | `ANTHROPIC_API_KEY` not set, or OAuth wasn't completed. Run `claude` once. |
| API returns 409 on stage call | Another stage is already running. Wait, or hit `/api/cancel`. |
| `Workspace missing train.py` | The folder you opened doesn't have a `train.py`. |
| Live chart stays empty during Stage 3/4 | `prober.py` isn't writing `.agent_probe/live/probe_live.json` — the chart will fill in once the run completes from `probe_result_N.json`. |
| Training crashes in a loop | Up to 5 auto-fix retries, then it stops. Check `agent.log` for the agent's reasoning. |

---

## License

See [LICENSE](LICENSE).
