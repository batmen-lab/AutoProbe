# Environment setup for the AutoProbe pipeline

Everything needed to get `make api` + `make web` serving and the `claude`
subprocesses working, on a fresh box.

**Scope.** This covers the *pipeline's own* environment: the Python venv, the
Node/web dependencies, the `claude` CLI and its auth, and verification. It
deliberately excludes two things:

- **Model routing via claude-code-router (ccr)** → [`CCR_and_openRouter.md`](CCR_and_openRouter.md).
  ccr is **optional**; [§6](#6-running-without-ccr) shows how to run without it,
  including the one Makefile trap that stops you.
- **Per-case / dataset setup** (`SUBPOP_DATA_DIR`, `subpopbench` downloads,
  credentialed medical data) → each case's `CASE.md` and
  [`CLAUDE_autoRun_for_subpop_cases_instruction.md`](CLAUDE_autoRun_for_subpop_cases_instruction.md).
  You can finish everything here and verify it without any dataset present.

---

## 0. What you are building

Three moving parts, all on localhost:

| Part | Port | Started by | What it is |
|---|---|---|---|
| FastAPI backend | `8765` | `make api` | Drives the pipeline; spawns `claude` subprocesses |
| Next.js frontend | `3000` | `make web` | The UI you click; talks to `:8765` |
| `claude` CLI | — | spawned per call | Every NLP and agent call is a fresh subprocess |

One venv at `venv/` runs **both** the API server and every workspace's
`train.py`. That is deliberate: `pipeline/stages.py::_train_interpreter()`
resolves `<repo>/venv/bin/python` explicitly rather than sniffing `PATH`, so it
holds no matter how you launched the server.

---

## 1. Prerequisites

| Need | Version | Check |
|---|---|---|
| Python | **3.12** exactly — not 3.13+ | `python3 --version` |
| Node.js | 18+ (22.x known good) | `node --version` |
| npm | 9+ | `npm --version` |
| `uv` | any recent | `uv --version` |
| `fuser` | any (`psmisc`) | `command -v fuser` |

### Why Python 3.12 and not newer

The SubpopBench case stack (`torch`, `torchvision`, `timm`, `netcal`) has no
3.13/3.14 wheels. **The system python is very likely too new** — on the
reference box `python3 --version` reports **3.14.4**, which cannot build this
stack at all. Do not try to make the system interpreter work; `uv` fetches a
private 3.12 for the venv and installs nothing system-wide.

### Install `uv` if missing

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# lands in ~/.local/bin — make sure that is on PATH
```

`make setup` falls back to `python3 -m venv` without `uv`, but on a box whose
`python3` is 3.13+ that fallback produces a venv the wheels won't install into.
**Install `uv`.**

`fuser` is used by `make api` to free port 8765 before binding. Without it that
line is a harmless no-op, but a stale server will block the port —
`apt-get install psmisc` if it is missing.

---

## 2. Python environment

### The one-liner

```bash
cd /path/to/AutoProbe
make setup
```

`make setup` does four things: creates `venv/` on Python 3.12 via `uv`,
installs `torch`/`torchvision` **from the correct wheel index first**, installs
`requirements.txt`, then runs `npm install` in `web/`.

### Choosing the torch wheel index — do this consciously

`make setup` defaults to the **CPU** index, because the PyPI default drags in
~2.5 GB of CUDA libraries a CPU box never loads:

```bash
make setup                                                    # CPU (default)
make setup PYTORCH_INDEX=https://download.pytorch.org/whl/cu126   # GPU box
```

**If the box has a GPU, pass the CUDA index.** Installing the CPU build on a
GPU box is silent — nothing errors, training just runs 10–50× slower on the
CPU. Verify after install ([§5](#5-verify)); `torch.cuda.is_available()` must
be `True`.

### Manual equivalent

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu      # or .../cu126
uv pip install --python venv/bin/python -r requirements.txt
```

Order matters: torch first from its own index, then `requirements.txt`.
`requirements.txt` deliberately **omits** torch/torchvision so that installing
it cannot pull the CUDA build from PyPI over your chosen wheels.

### What gets installed, and the one pin that matters

`requirements.txt` covers three groups — API (`fastapi`, `uvicorn[standard]`,
`pydantic`), the training stack (`numpy`, `pandas`, `scikit-learn`, `scipy`,
`tqdm`, `plotly`, `kaleido`, `matplotlib`), and the case stack (`timm`,
`netcal`, `pillow`).

> **`transformers<5` is pinned and must stay pinned.** v5 dropped tokenizer
> aliases the vendored SubpopBench sources import. (Separately,
> `transformers.AdamW` was already removed in 4.5x — the case sources now
> import `torch.optim.AdamW`, which is the identical algorithm.)

Reference versions from a known-good box (Python 3.12.14):

```
torch 2.11.0+cu128   torchvision 0.26.0+cu128   transformers 4.57.6
fastapi 0.141.1      uvicorn 0.52.4             pydantic 2.13.5
numpy 2.5.2          pandas 3.0.5               scikit-learn 1.9.0
scipy 1.18.1         timm 1.0.29                netcal 1.4.0
plotly 7.0.0         matplotlib 3.11.1          pillow 12.3.0
```

### Adding packages later

Install into the **same** venv — the pipeline runs `train.py` with
`venv/bin/python`:

```bash
uv pip install --python venv/bin/python <package>
```

If a workspace needs a stack that genuinely conflicts, point it elsewhere per
case instead of polluting the shared venv:

```bash
AUTOPROBE_TRAIN_PYTHON=/path/to/other/venv/bin/python make api
```

---

## 3. Node / web dependencies

```bash
cd web && npm install && cd ..
```

(`make setup` already did this.) The frontend is Next.js 15 + React 19 +
Tailwind 3.4, dev server on port 3000, hardcoded in `web/package.json`
(`next dev -p 3000`). `node_modules/` is gitignored; re-run `npm install` after
a fresh clone.

---

## 4. The `claude` CLI and auth

Every NLP and agent call in the pipeline is a `claude` subprocess, so the CLI
must be installed **and authenticated** as your shell user.

```bash
npm install -g @anthropic-ai/claude-code
claude --version        # e.g. 2.1.258 (Claude Code)
```

Authenticate one of two ways — both work for NLP and agent calls alike:

**A — API key.** From <https://console.anthropic.com/>:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc && source ~/.bashrc
```

**B — interactive OAuth (Pro/Max).** Run `claude` once, complete the browser
flow, `Ctrl-C`:

```bash
claude
```

> **Auth must belong to the user that runs the server.** The pipeline spawns
> `claude` as a child of the API process; credentials in another user's home
> or in a shell you did not launch the server from will not be seen.

---

## 5. Verify

### `make doctor` — read every line

```bash
make doctor
```

Expected on a healthy GPU box:

```
python   : Python 3.12.14  (venv/bin/python)
train py : /path/to/AutoProbe/venv/bin/python
node     : v22.23.2
claude   : 2.1.258 (Claude Code)
ccr      : installed            <- "NOT FOUND" is fine, see §6
torch 2.11.0+cu128  transformers 4.57.6
```

- `python` **must** say 3.12.x. Anything else and the venv is wrong.
- `train py` is the interpreter that will run every `train.py`. If it is not
  `<repo>/venv/bin/python`, something overrode `AUTOPROBE_TRAIN_PYTHON`.
- `claude : NOT FOUND` → [§4](#4-the-claude-cli-and-auth).
- The `+cu128` suffix is your GPU tell. A bare `2.11.0` means the CPU build.

### GPU check — `doctor` does not do this

```bash
venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# -> True NVIDIA L4
```

Do not trust prose about this box's hardware — `CLAUDE.md` still says "this box
has no GPU", which is **stale**; the reference box has an NVIDIA L4 and
`torch.cuda.is_available()` is `True`. Always check.

### Import check

```bash
venv/bin/python -c "import fastapi, uvicorn, pydantic, torch, torchvision, transformers, sklearn, scipy, pandas, timm, netcal, PIL, plotly, matplotlib; print('all imports OK')"
```

### End-to-end LLM smoke test

```bash
venv/bin/python test.py
```

It runs the same call shapes the pipeline uses, through the same subprocess
environment `pipeline/llm.py` builds:

```
── NLP model (Claude, no tools) ────────  PASS
── Agent (Claude, full tools) ──────────  PASS
── Web search (informational) ──────────  FAIL   <- expected when routed
```

**The first two must pass.** The third is informational: `WebSearch` is
Anthropic's server-side search and returns nothing through a third-party
provider, which is exactly why `pipeline/llm.py` leaves it out of `NLP_TOOLS`.
Expect it to fail on a routed setup; ignore it.

---

## 6. Running without ccr

ccr is optional. Skipping it means the CLI talks to Anthropic directly with the
auth from [§4](#4-the-claude-cli-and-auth).

> **The trap: `make api` will not start without ccr installed.** In the
> Makefile the target is `api: ccr-up`, and `ccr-up` begins with
> `command -v ccr >/dev/null || { echo "ERROR: ccr not installed"; exit 1; }`.
> Make aborts a target when a prerequisite fails, so on a box without ccr
> `make api` dies before the server is ever launched. `AUTOPROBE_ROUTER=off`
> does **not** rescue this — it disables env *injection* inside
> `pipeline/router.py`, it does not remove the Makefile prerequisite. The
> README's `AUTOPROBE_ROUTER=off make api` line is therefore misleading on a
> ccr-less box.

Two ways through:

**Bypass the Makefile (recommended, nothing to install):**

```bash
AUTOPROBE_ROUTER=off venv/bin/python -m server.app     # terminal 1, :8765
make web                                               # terminal 2, :3000
```

That is byte-for-byte what `make api` runs after its ccr check. `make web` has
no ccr prerequisite and is always safe.

**Or install ccr and leave it idle** — `make api` then satisfies its check and
`AUTOPROBE_ROUTER=off` keeps traffic going straight to Anthropic.

Recognised off-switch values (`pipeline/router.py`): `off`, `none`, `0`,
`direct`. Any `ANTHROPIC_*` variable you export yourself always wins over what
the router reads.

---

## 7. Running

Two terminals from the repo root:

```bash
make api      # terminal 1 — Uvicorn on http://127.0.0.1:8765
make web      # terminal 2 — Next.js on http://localhost:3000
```

Then open <http://localhost:3000>.

### Backgrounding them (agents, headless boxes)

```bash
( cd /path/to/AutoProbe && nohup make api > /tmp/api.log 2>&1 & )
( cd /path/to/AutoProbe && nohup make web > /tmp/web.log 2>&1 & )
```

> **Keep each `cd` inside its own subshell.** `cd X && nohup make api ... &`
> backgrounds the *whole* `cd && make`, so the `cd` never applies to your
> shell, and a following bare `make web` runs in `$HOME` and dies with
> `No rule to make target 'web'`.

Readiness (do not just sleep):

```bash
until curl -s --max-time 2 http://127.0.0.1:8765/api/health >/dev/null; do sleep 2; done
curl -s http://127.0.0.1:8765/api/health          # {"ok":true,"busy":false}
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/   # 200
```

`make api` runs `fuser -k -TERM 8765/tcp` first, so restarting it reclaims the
port from a previous server automatically.

### Other entry points

- `make api-codex` — same server on **:8766** with `LLM_BACKEND=codex`, driving
  the `codex` CLI instead of `claude`. Needs `@openai/codex` installed and
  `codex login`. Runs land under `response_codex/`. No ccr involvement.
- `python main.py` (`make cli`) — legacy CLI driver. **Its Stage-4 loop is
  disabled on purpose** and raises `SystemExit`. Do not use it to drive a run;
  see `CLAUDE.md`.

### Ports at a glance

| Port | Service | Override |
|---|---|---|
| 8765 | FastAPI | `API_PORT=… venv/bin/python -m server.app` |
| 3000 | Next.js | edit `web/package.json` |
| 8766 | FastAPI (codex) | `make api-codex` |
| 3456 | ccr gateway | out of scope — see `CCR_and_openRouter.md` |

---

## 8. Environment variables (non-ccr)

| Variable | Default | Effect |
|---|---|---|
| `AUTOPROBE_TRAIN_PYTHON` | `<repo>/venv/bin/python` | Interpreter used to run each workspace's `train.py`. |
| `API_PORT` | `8765` | Port `server.app` binds. |
| `LLM_BACKEND` | `claude` | `codex` swaps in `pipeline/llm_codex.py`; runs go to `response_codex/`. |
| `AUTOPROBE_ROUTER` | `ccr` | `off`/`none`/`0`/`direct` disables ccr env injection. |
| `ANTHROPIC_API_KEY` | — | Used by the `claude` CLI. Anything you export wins over the router. |

---

## 9. State on disk

| Path | Gitignored | What |
|---|---|---|
| `venv/` | yes | The one interpreter for server *and* `train.py` |
| `web/node_modules/` | yes | Frontend deps |
| `response/` | yes | Per-run metadata: `stage.json`, `agent.log`, stage artifacts |
| `response/_app_state.json` | yes | Current + recent workspaces (VS Code-style) |
| `response_codex/` | yes | Same, for `LLM_BACKEND=codex` |

`response/` is the audit trail and survives workspace cleanup — do not delete it
to "clean up" while runs matter. Nothing in this section needs to exist before
first start; the server creates it.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `make api` → `ERROR: ccr not installed` | `api` depends on `ccr-up` | [§6](#6-running-without-ccr) — run `venv/bin/python -m server.app` directly |
| `ModuleNotFoundError` running `train.py` | Wrong interpreter (system python, not the venv) | `make doctor`; check `train py`; unset a stray `AUTOPROBE_TRAIN_PYTHON` |
| torch wheels won't resolve during setup | venv built on 3.13/3.14 | Delete `venv/`, install `uv`, `make setup` again |
| Training is inexplicably slow | CPU torch on a GPU box | `torch.__version__` lacks `+cuXXX` → reinstall with the CUDA index |
| `ImportError` on a tokenizer alias / `transformers.AdamW` | `transformers>=5` installed | Re-pin `transformers<5` |
| `make web` → `No rule to make target 'web'` | Running outside the repo root (backgrounding `cd` trap) | [§7](#7-running) — wrap each `cd` in its own subshell |
| `Failed to fetch` / UI can't reach API | API not up, or port 8765 taken | `curl :8765/api/health`; `fuser -k 8765/tcp` |
| `409` from an API call | Another long stage is running — one `asyncio.Lock` covers them all | Wait, or `POST /api/cancel` |
| `test.py` NLP/agent tests fail | CLI not authenticated as the server's user | [§4](#4-the-claude-cli-and-auth) |
| `test.py` web-search test fails | Expected on a routed setup | Ignore |
| Port 3000 in use | Old `next dev` alive | `fuser -k 3000/tcp` |

---

## 11. Fresh-box checklist

```bash
# 0. prerequisites
uv --version && node --version && npm --version

# 1. clone
git clone <your-fork-of-AutoProbe>.git && cd AutoProbe

# 2. python + node deps  (add PYTORCH_INDEX=.../cu126 on a GPU box)
make setup

# 3. claude CLI + auth
npm install -g @anthropic-ai/claude-code && claude --version
export ANTHROPIC_API_KEY=sk-ant-...        # or run `claude` once for OAuth

# 4. verify
make doctor
venv/bin/python -c "import torch;print('cuda:',torch.cuda.is_available())"
venv/bin/python test.py                    # first two tests must PASS

# 5. run  (without ccr, use: AUTOPROBE_ROUTER=off venv/bin/python -m server.app)
make api        # terminal 1
make web        # terminal 2  -> http://localhost:3000
```

At this point the pipeline is ready. Datasets and per-case configuration are a
separate step — see each case's `CASE.md` and
[`CLAUDE_autoRun_for_subpop_cases_instruction.md`](CLAUDE_autoRun_for_subpop_cases_instruction.md).
