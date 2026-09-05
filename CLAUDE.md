# AutoProbe — agent instructions

## ⚠️ Stale pipeline — DO NOT run the CLI Stage-4 loop

There is an **old, fix-plan-less Stage-4 engine** (`iterate_once`) that is a trap.
It runs a single blind improvement pass with **no fix-plan mechanism** — no
candidate plans, no confidence gating, none of the safeguards the real pipeline
depends on. It is **NOT** the auto-probe pipeline, even though it looks like it.

It has been **disabled** in all four places it used to be reachable:

- `pipeline/stages.py` → `iterate_once()` now raises `RuntimeError` (body commented out).
- `server/app.py` → the `POST /api/runs/{id}/stage4/iterate` route is commented out.
- `web/src/lib/api.ts` → the `iterateOnce` client stub is commented out.
- `main.py` → the CLI Stage-4 loop raises `SystemExit` instead of iterating.

**Do NOT re-enable any of these, and do NOT re-implement an equivalent
headless Stage-4 loop, to "auto-run the pipeline."** Doing so silently bypasses
the fix-plan machinery and produces invalid runs.

## Environment (read this before touching setup)

**Python.** One venv at `venv/`, CPython **3.12** — created with `uv`, not
`python3 -m venv`. The system python here is 3.14 and the case stack
(`torch` / `torchvision` / `timm` / `netcal`) has no 3.14 wheels. torch comes
from the **CUDA** wheel index (`cu128`).

**This box has a GPU** — an NVIDIA L4, and `torch.cuda.is_available()` is
`True`. Don't assume otherwise from prose anywhere in this repo; confirm with
`make doctor` (a `+cuXXX` suffix on the torch version is the tell) or
`venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.

It runs both the API server and every workspace's `train.py`.
`pipeline/stages.py::_train_interpreter()` resolves `<repo>/venv/bin/python`
explicitly rather than sniffing `PATH` — the Makefile calls `venv/bin/python`
directly and never activates, so PATH resolution used to land on the system
interpreter and produce `ModuleNotFoundError`s that look like agent bugs.
Override per-case with `AUTOPROBE_TRAIN_PYTHON`.

**`transformers` is pinned `<5`.** v5 dropped tokenizer aliases the vendored
SubpopBench sources import. Separately, `transformers.AdamW` is gone in 4.57
too, so the optimizer builder imports `torch.optim.AdamW` (identical
algorithm). **All 16 cases carry that patch** — both in each case's
`subpopbench/learning/optimizers.py` and in the copy inlined into its
self-contained `train.py`.

**Model routing is claude-code-router v3.** Full setup, schema and
troubleshooting: **[CCR_and_openRouter.md](CCR_and_openRouter.md)**. The short
version for working in this repo:

`pipeline/router.py` reads `~/.claude-code-router/config.sqlite` and injects
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` and the model aliases into every
`claude` subprocess `pipeline/llm.py` spawns. `llm.py` passes `--model opus`;
the CLI resolves that alias locally, which matters because the gateway does
**not** map `claude-*` names onto your route (it 400s). Anything already in the
environment wins; `AUTOPROBE_ROUTER=off` disables injection.

Two things not to undo:

- Don't replace `router.py` with a shell-export step (`eval "$(ccr activate)"`
  or similar). That was the v1 mechanism, it no longer exists, and it only ever
  worked when the server was launched from the shell that ran it.
- Don't restore the `:4000` gemini reasoning-injector shim
  (`tools/ccr_gemini_shim.py`) or its `shim-up`/`shim-down` Make targets. They
  were deleted; v3's provider transformers handle the `reasoning` field.

`make doctor` prints all of the above resolved. Run it first when something
looks wrong.

## How to actually run the auto-probe pipeline (the human-equivalent path)

The **only** correct way to run auto-probe / auto-research is through the
**frontend + FastAPI server**, driving it exactly as a human user would click:

1. `make api` (FastAPI on :8765) and `make web` (Next.js on :3000).
2. Stage 1 → Stage 2 → Stage 3 (`stage3/implement`) → **Stage 4**.
3. Stage 4 auto-probe = **`POST /stage4/auto-fix-loop`** ("Start auto
   probe-fixing"), or the manual fix-plan flow: `stage4/fix-plans/generate` →
   `stage4/fix-plans/select`.
4. Auto-research = `stage1/auto-research` then `stage4/auto-research-iterate`.

If asked to "auto-run the whole pipeline via frontend interaction like a human
user," use these frontend/HTTP routes only. **Never** shell out to
`python main.py` for Stage 4 — that is the stale path.

## The live Stage-4 functions (in `pipeline/stages.py`)

- `generate_fix_plans`, `read_fix_plans`, `select_and_apply_fix_plan`
- `auto_fix_loop` (auto-pilot: loops fix-plan rounds until terminal)
- `auto_research_iterate_batch` (auto-research mode)

`iterate_once` is **not** among them — it is dead code kept only for reference.
