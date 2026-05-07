# Agentic Probe Rein

An agentic pipeline that automatically designs, implements, and iteratively improves evaluation probes for ML/DL training pipelines. Given a project description, it generates quantitative probes grounded in peer literature, implements them as runnable code, and uses AI agents to drive a training metric toward a target threshold.

---

## How It Works

The system chains two types of AI calls:

- **NLP calls** — Claude (no tools, no session) for structured JSON generation: probe design, confidence scoring, and implementation planning.
- **Agent calls** — Claude Code CLI (full tools, full filesystem access) for code generation, code modification, crash fixing, and iterative improvement.

Each call is fully isolated: no shared history, no persistent context.

---

## Workflow

```
User describes project
        │
        ▼
[NLP] Generate 10 probes          ← PROMPT_ONE  (probe angles from peer literature)
        │
        ▼
[NLP] Score probe confidence      ← PROMPT_TWO  (source verification, 0.0–1.0)
        │
        ▼
User selects a probe (1–10)
        │
        ▼
[NLP] Generate 3 dev plans        ← PROMPT_THREE (metric + threshold + implementation steps)
        │
        ▼
[NLP] Score plan practicality     ← PROMPT_FOUR  (engineering feasibility, 0.0–1.0)
        │
        ▼
User selects a dev plan (1–3)
        │
        ▼
[Agent] Implement probe           ← PROMPT_FIVE  (writes prober.py, integrates into train.py)
        │
        ▼
[Run train.py] ──crash──► [Agent] Fix  ← PROMPT_EIGHT  (exception catcher, up to 5 retries)
        │ success
        ▼
(Optional) Auto-research feature (default: off)
        │  enabled
        ├──► [Agent] Annotate train.py  ← PROMPT_SIX  (10 targeted improvement comments)
        │           │
        │    [Run + fix if crashed]
        │
        ▼
User sets iteration count (default: 3)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Snapshot train.py                              │
│  [Agent] Improve training pipeline              │  ← PROMPT_SEVEN
│  [Run train.py + fix if crashed]               │  ← PROMPT_EIGHT
│  Check probe_result_N.json → status == "PASS"? │
│  If yes: stop early. If no: repeat N times.     │
└─────────────────────────────────────────────────┘
        │
        ▼
User chooses: try another probe or exit
```

### What each probe run produces

After each successful `train.py` run the probe writes two artifacts into the workspace:

| Artifact | Path | Contents |
|---|---|---|
| Metric JSON | `.agent_probe/metric/probe_result_N.json` | metric name, per-epoch values, min/max/mean/std, delta, threshold, status (PASS/FAIL), conclusion |
| Plot PDF | `.agent_probe/plot/probe_result_N.pdf` | Plotly line chart: metric over epochs, dashed threshold line, crossing annotation, stats box, consistent axis range across runs |

Additional files created automatically:

| Path | Contents |
|---|---|
| `.agent_probe/snapshot/train_version_N.py` | Snapshot of `train.py` before each agent iteration (for revert) |
| `.agent_probe/change_log_N.txt` | Log of changes made in each iteration |
| `.agent_probe/_axis_range.json` | Cached chart axis ranges for cross-run visual consistency |

---

## Project Structure

```
agentic_probe_rein/
├── main.py                         # Entry point — full workflow orchestration
├── Questions.py                    # All user-facing prompt strings
├── test.py                         # Smoke tests (NLP call, agent call, web search)
│
├── hard_prompt/                    # System prompts (one file per agent)
│   ├── nlp_prober_gen.py           #  PROMPT_ONE   — generate 10 probe designs
│   ├── nlp_prober_confi_comput.py  #  PROMPT_TWO   — score probe confidence
│   ├── nlp_dev_doc_gen.py          #  PROMPT_THREE — generate 3 implementation plans
│   ├── nlp_dd_confi_comput.py      #  PROMPT_FOUR  — score plan practicality
│   ├── agent_dd_implement.py       #  PROMPT_FIVE  — implement prober.py + integrate
│   ├── agent_improve_commentor.py  #  PROMPT_SIX   — annotate train.py (10 comments)
│   ├── agent_iterat_improver.py    #  PROMPT_SEVEN — iteratively improve train.py
│   └── agent_exception_catcher.py  #  PROMPT_EIGHT — fix crashed train.py
│
├── response/                       # Run outputs (timestamped, auto-created)
│   └── YYYYMMDDHHMMSS/
│       ├── probe_designs.json
│       ├── probe_confidenced.json
│       ├── dev_doc.json
│       ├── dev_doc_confidenced.json
│       └── progressbar.json        # Resume state
│
├── mimic/                          # Example target workspace (MIMIC-III mortality)
│   ├── train.py                    # Training loop (modified by agents)
│   ├── prober.py                   # Equalized-odds fairness probe (agent-generated)
│   ├── dataset.py                  # PyTorch dataset loader
│   ├── preprocess.py               # TF-IDF feature extraction
│   └── .agent_probe/               # Auto-created probe output directory
│
└── dummy_project/                  # Minimal example workspace for testing
    ├── train.py
    └── data_process.py
```

---

## Setup

### 1. Python

Python 3.10 or later is required.

```bash
python --version   # must be 3.10+
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or: venv\Scripts\activate     # Windows
```

### 2. Python dependencies

The orchestration code itself only uses the standard library. The **target workspace** (the ML project being probed) needs its own dependencies — for the included `mimic/` example:

```bash
pip install torch torchvision scipy scikit-learn numpy plotly kaleido matplotlib
```

For your own project, install whatever that project requires.

> `kaleido` is the Plotly static image export backend. If it is unavailable, the probe falls back to matplotlib; if that also fails, it writes a placeholder PDF.

### 3. Node.js and Claude Code CLI

The agent calls use the **Claude Code CLI** (`claude`), not the Python SDK. Node.js 18+ is required.

```bash
# Install Node.js (if not already installed)
# Ubuntu / Debian:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS (Homebrew):
brew install node

# Verify:
node --version    # 18+
npm --version
```

Install Claude Code globally:

```bash
npm install -g @anthropic-ai/claude-code
claude --version  # verify install
```

### 4. API keys

Two API keys are required — one for each call type:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Claude Code CLI (PROMPT_FIVE through EIGHT)
```

The NLP calls (PROMPT_ONE through FOUR) also go through the Claude Code CLI (the `claude -p --tools ""` invocation). Both call types use the same `ANTHROPIC_API_KEY`.

Put these in your shell profile (`~/.bashrc`, `~/.zshrc`) to avoid re-exporting each session:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

### 5. Authenticate Claude Code

On first use, Claude Code requires a one-time authentication:

```bash
claude
# Follow the browser-based OAuth flow, then Ctrl+C to exit the interactive session
```

### 6. Point the tool at your project

Open [main.py](main.py) and set `WROKING_SPACE` to the absolute path of your ML project directory:

```python
WROKING_SPACE: Final = "/absolute/path/to/your/project"
```

Your project directory must contain a `train.py` that:
- Runs a complete training loop when executed with `python train.py`
- Exits with code 0 on success
- Exits with a non-zero code and prints a traceback on failure

The agent reads your existing code before making any changes.

### 7. Configure models

In [main.py](main.py), adjust the model names if needed:

```python
NLP_MODEL   = "opus"   # used for probe/plan generation (--tools "" mode)
AGENT_MODEL = "opus"   # used for code generation and improvement (full tools mode)
```

These are the `--model` values passed to the `claude` CLI. Any model ID supported by Claude Code works here (e.g., `"sonnet"`, `"haiku"`, `"claude-opus-4-7"`).

---

## Running

### Smoke test (optional but recommended)

Verify that both call types are working before a full run:

```bash
python test.py
```

All three checks should print `PASS`:

```
── NLP model (Claude, no tools) ────────
  PASS — got: {'status': 'ok', 'model': 'nlp'}
── Agent (Claude, full tools) ──────────
  PASS — got: 'PONG'
── Web search (NLP, CRWV stock price) ──
  PASS — CRWV price: 123.45 (source: ...)
```

### Start

```bash
python main.py
```

The session is **stateful**. Each run creates a timestamped directory under `response/`. If you quit mid-run, re-running `main.py` offers to resume from where you left off.

### Interactive prompts

| Step | Prompt | Input |
|---|---|---|
| 0 | Confirm dependencies installed | Y / N |
| 1 | Describe your ML project and dataset | Free text |
| 2 | Select a probe design | 1–10 |
| 3 | Select an implementation plan | 1–3 |
| 4 | Enable auto-research annotation | Y / N (default N) |
| 5 | Number of improvement iterations | Integer (default 3) |
| 6 | Try another probe or exit | Y / N |

---

## Key Design Decisions

- **Supervisor-scored confidence** — PROMPT_TWO and PROMPT_FOUR are separate agents from the generators. They fill in the confidence field independently, avoiding self-assessment bias.
- **Isolated calls** — every NLP call uses `--no-session-persistence`; every agent call is a new subprocess. No shared state between calls.
- **Frozen probe definition** — once `prober.py` is written (PROMPT_FIVE), the improvement agent (PROMPT_SEVEN) is instructed never to modify it. Only `train.py` and supporting files are changed.
- **Snapshot before each iteration** — `train.py` is saved to `.agent_probe/snapshot/train_version_N.py` before every agent modification, enabling manual revert.
- **Change log trail** — each iteration writes `.agent_probe/change_log_N.txt` so the next iteration can see what was already tried and avoid repeating failed approaches.
- **Exception catcher cap** — up to 5 auto-fix retries per run. If all fail, the error is surfaced and execution halts.
- **Early stopping** — iterative improvement stops as soon as `probe_result_N.json` reports `"status": "PASS"`, regardless of the configured iteration count.
- **Consistent chart axes** — the first probe run caches the y-range in `_axis_range.json`; subsequent runs reuse it so plots are visually comparable across iterations.
