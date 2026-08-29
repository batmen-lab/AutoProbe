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
