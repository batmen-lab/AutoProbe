"""FastAPI server fronting the pipeline.

Single-active-stage discipline: only one long-running stage action (3 or 4)
may be in flight across all runs. Concurrent calls 409.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline import (
    RunState,
    Stage,
    list_runs,
    load_run,
    new_run,
    list_recent_workspaces,
    open_workspace,
    is_valid_workspace,
)
from pipeline.workspace import current_workspace
from pipeline.state import (
    PROBE_CONFIDENCED,
    DEV_DOC_CONFIDENCED,
)
from pipeline import stages as stages_mod
from pipeline.llm import cancel_current


app = FastAPI(title="Agentic Probe — Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Single-active-stage gate ─────────────────────────────────────────────────
# Long stage actions (probe/plan generation, implement, iterate, fix-loops) run
# for many minutes. We do NOT hold the HTTP request open for their duration:
# a browser fetch that timed out mid-action used to cancel the request and
# strand the run at phase="running" (the "Failed to fetch" symptom). Instead an
# action is launched as a DETACHED background task — the POST returns the
# current state immediately and the frontend polls /api/runs/{id} for progress.
# The task ALWAYS finalizes state (even on crash) so a run can never be wedged.
_busy = False
_current_run_id: str | None = None
_last_error: dict[str, str] = {}  # run_id -> last failure message (UI surfaces it)
_cancelled: set[str] = set()      # runs whose action was intentionally cancelled


def _safe_phase(state: RunState) -> str:
    """Where to park a run whose action ended unexpectedly (mirrors cancel())."""
    return "ready" if state.record.stage == 4 else "input"


async def _run_action(fn, args: tuple, run_id: str) -> None:
    """Body of a background stage action. Runs the blocking fn in a thread and,
    no matter how it ends, leaves the run in a consistent, non-"running" state."""
    global _busy, _current_run_id
    try:
        await asyncio.to_thread(fn, *args)
    except Exception as e:  # noqa: BLE001 — a failure must never strand the run
        try:
            st = load_run(run_id)
            st.set_action(None)
            st.set_phase(st.record.stage, _safe_phase(st))
        except Exception:
            pass
        if run_id not in _cancelled:
            _last_error[run_id] = f"{type(e).__name__}: {e}"
    finally:
        _busy = False
        _current_run_id = None
        _cancelled.discard(run_id)


def _launch(fn, *args, run_id: str) -> None:
    """Start a long stage action off the request path. 409 if one is running.

    `_busy` is set synchronously here (before the route returns) so a second
    concurrent request reliably 409s instead of racing into the background.
    """
    global _busy, _current_run_id
    if _busy:
        raise HTTPException(409, "Another stage action is already running.")
    _busy = True
    _current_run_id = run_id
    _last_error.pop(run_id, None)
    asyncio.create_task(_run_action(fn, args, run_id))


# ── Models ───────────────────────────────────────────────────────────────────
class OpenWorkspaceBody(BaseModel):
    path: str


class NewRunBody(BaseModel):
    workspace: str | None = None  # default: current workspace


class ContextBody(BaseModel):
    context: str


class SelectBody(BaseModel):
    index: int  # 1-based


class RevertBody(BaseModel):
    to_stage: int
    keep_workspace: bool = False  # post-PASS "keep changes" path (target=1 only)


class ListDirBody(BaseModel):
    path: str


class AutoResearchIterateBody(BaseModel):
    count: int  # number of auto-research rounds to run in this batch


class FixPlanGenerateBody(BaseModel):
    hint: str | None = None


# ── Workspace ────────────────────────────────────────────────────────────────
@app.get("/api/workspace")
def get_workspace():
    s = list_recent_workspaces()
    return {
        "current": s.current_workspace,
        "recent": s.recent_workspaces,
    }


@app.post("/api/workspace/open")
def open_workspace_route(body: OpenWorkspaceBody):
    try:
        s = open_workspace(body.path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"current": s.current_workspace, "recent": s.recent_workspaces}


@app.post("/api/workspace/browse")
def browse_dir(body: ListDirBody):
    """List subdirectories of a path — for the 'Open Folder' UX.

    Returns siblings + parent so the frontend can show a tree-ish picker.
    """
    p = Path(body.path).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {p}")
    try:
        entries = sorted(
            [
                {
                    "name": child.name,
                    "path": str(child),
                    "is_workspace": is_valid_workspace(child),
                }
                for child in p.iterdir()
                if child.is_dir() and not child.name.startswith(".")
            ],
            key=lambda x: x["name"].lower(),
        )
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {p}")
    return {
        "path": str(p),
        "parent": str(p.parent) if p.parent != p else None,
        "is_workspace": is_valid_workspace(p),
        "entries": entries,
    }


# ── Runs ─────────────────────────────────────────────────────────────────────
@app.get("/api/runs")
def list_runs_route(workspace: str | None = Query(None)):
    ws = Path(workspace) if workspace else None
    return {"runs": list_runs(workspace=ws)}


@app.post("/api/runs")
def create_run(body: NewRunBody):
    ws = body.workspace
    if ws is None:
        try:
            ws_path = current_workspace()
        except RuntimeError as e:
            raise HTTPException(400, str(e))
    else:
        ws_path = Path(ws)
    try:
        state = new_run(ws_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _state_payload(state)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        state = load_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return _state_payload(state)


# ── Stage 1 ──────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/stage1/context")
def set_context(run_id: str, body: ContextBody):
    state = load_run(run_id)
    state.set_context(body.context)
    return _state_payload(state)


@app.post("/api/runs/{run_id}/stage1/generate")
async def stage1_generate(run_id: str):
    state = load_run(run_id)
    _launch(stages_mod.generate_probes, state, run_id=run_id)
    return _stage1_artifact(load_run(run_id))  # null until the task completes


@app.post("/api/runs/{run_id}/stage1/select")
def stage1_select(run_id: str, body: SelectBody):
    state = load_run(run_id)
    try:
        stages_mod.select_probe(state, body.index)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _state_payload(state)


@app.get("/api/runs/{run_id}/stage1/artifact")
def stage1_artifact(run_id: str):
    state = load_run(run_id)
    return _stage1_artifact(state)


@app.post("/api/runs/{run_id}/stage1/auto-research")
async def stage1_auto_research(run_id: str):
    """Skip NLP probe + dev-plan, jump straight to a performance-monitoring probe.

    The agent picks a metric, writes prober.py, integrates train.py, then we run
    training + commentor + training. After this the run is parked at stage 4.
    """
    state = load_run(run_id)
    _launch(stages_mod.auto_research_setup, state, run_id=run_id)
    return _state_payload(load_run(run_id))


# ── Stage 2 ──────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/stage2/generate")
async def stage2_generate(run_id: str):
    state = load_run(run_id)
    _launch(stages_mod.generate_dev_plans, state, run_id=run_id)
    return _stage2_artifact(load_run(run_id))  # null until the task completes


@app.post("/api/runs/{run_id}/stage2/select")
def stage2_select(run_id: str, body: SelectBody):
    state = load_run(run_id)
    try:
        stages_mod.select_plan(state, body.index)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _state_payload(state)


@app.get("/api/runs/{run_id}/stage2/artifact")
def stage2_artifact(run_id: str):
    state = load_run(run_id)
    return _stage2_artifact(state)


# ── Stage 3 ──────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/stage3/implement")
async def stage3_implement(run_id: str):
    state = load_run(run_id)
    _launch(stages_mod.implement, state, run_id=run_id)
    return _state_payload(load_run(run_id))


# ── Stage 4 ──────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/stage4/iterate")
async def stage4_iterate(run_id: str):
    state = load_run(run_id)
    _launch(stages_mod.iterate_once, state, run_id=run_id)
    return _state_payload(load_run(run_id))


@app.post("/api/runs/{run_id}/stage4/auto-fix-loop")
async def stage4_auto_fix_loop(run_id: str):
    """Stage-4 auto-pilot: run fix-plan rounds (auto-pick highest confidence)
    until a terminal state (PASS / best-effort / stagnant). Drives the
    "Start auto probe-fixing" button on the frontend.
    """
    state = load_run(run_id)
    # Failures (ValueError / agent didn't write fix-plans / etc.) are caught in
    # _run_action: it parks the run at phase=ready and records `last_error`,
    # which the frontend surfaces via polling.
    _launch(stages_mod.auto_fix_loop, state, run_id=run_id)
    return _state_payload(load_run(run_id))


@app.post("/api/runs/{run_id}/stage4/fix-plans/generate")
async def stage4_fix_plans_generate(
    run_id: str,
    body: FixPlanGenerateBody | None = None,
):
    """Generate 3 candidate fix plans for the next iteration. Triggered after
    a FAIL round when the user clicks "Continue fixing" → "Generate fixing
    plans". Optional `hint` body parameter is passed to the generator agent
    as user-provided direction (non-binding). Returns the JSON list so the
    UI can render cards.
    """
    state = load_run(run_id)
    hint = body.hint if body is not None else None
    _launch(stages_mod.generate_fix_plans, state, hint, run_id=run_id)
    # Fix plans aren't ready yet — the UI polls /stage4/fix-plans/artifact.
    return {"state": _state_payload(load_run(run_id)), **stages_mod.read_fix_plans(load_run(run_id))}


@app.get("/api/runs/{run_id}/stage4/fix-plans/artifact")
def stage4_fix_plans_artifact(run_id: str):
    """Return the open fix-plan set (if any). Used by the UI to re-render
    after a page refresh or polling tick."""
    state = load_run(run_id)
    return stages_mod.read_fix_plans(state)


@app.post("/api/runs/{run_id}/stage4/fix-plans/select")
async def stage4_fix_plans_select(run_id: str, body: SelectBody):
    """User picked one of the 3 fix plans — apply it + run training."""
    state = load_run(run_id)
    _launch(stages_mod.select_and_apply_fix_plan, state, body.index, run_id=run_id)
    return _state_payload(load_run(run_id))


@app.post("/api/runs/{run_id}/stage4/auto-research-iterate")
async def stage4_auto_research_iterate(run_id: str, body: AutoResearchIterateBody):
    """Run a batch of `count` auto-research rounds with revert-on-regression."""
    if body.count <= 0 or body.count > 100:
        raise HTTPException(400, "count must be between 1 and 100")
    state = load_run(run_id)
    _launch(stages_mod.auto_research_iterate_batch, state, body.count, run_id=run_id)
    return _state_payload(load_run(run_id))


# ── Revert ───────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/revert")
def revert(run_id: str, body: RevertBody):
    state = load_run(run_id)
    try:
        result = state.revert_to(body.to_stage, keep_workspace=body.keep_workspace)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"result": result, "state": _state_payload(state)}


# ── Log streaming (SSE) ──────────────────────────────────────────────────────
@app.get("/api/runs/{run_id}/log/stream")
async def stream_log(run_id: str):
    state = load_run(run_id)
    log_path = state.log_path

    async def gen():
        # Tail new appends. Buffer partial trailing lines so we don't emit a
        # half-written log line as if it were a complete one.
        last_size = 0
        buf = b""
        while True:
            try:
                if log_path.exists():
                    size = log_path.stat().st_size
                    if size > last_size:
                        with log_path.open("rb") as f:
                            f.seek(last_size)
                            chunk = f.read(size - last_size)
                        last_size = size
                        buf += chunk
                        if b"\n" in buf:
                            parts = buf.split(b"\n")
                            buf = parts.pop()  # keep trailing partial line
                            for line in parts:
                                yield f"data: {line.decode('utf-8', errors='replace')}\n\n"
                yield ": ping\n\n"
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/log")
def read_log(run_id: str):
    state = load_run(run_id)
    if not state.log_path.exists():
        return {"log": ""}
    return {"log": state.log_path.read_text(errors="replace")}


# ── Live metric (per-epoch, dynamic during training) ─────────────────────────
@app.get("/api/runs/{run_id}/live-metric")
def live_metric(run_id: str):
    """Return the current run's per-epoch metric trajectory for live charting.

    Reads `<workspace>/.agent_probe/live/probe_live.json` if the agent emitted
    it (from the prompt). Falls back to the last-completed
    `.agent_probe/metric/probe_result_N.json` so we can still display data after
    a run finishes.
    """
    state = load_run(run_id)
    live = state.workspace / ".agent_probe" / "live" / "probe_live.json"
    if live.exists():
        try:
            return {"source": "live", **json.loads(live.read_text())}
        except Exception:
            pass

    # Fallback: latest completed result.
    metric_dir = state.workspace / ".agent_probe" / "metric"
    if not metric_dir.exists():
        return {"source": "none", "values": []}
    nums: list[int] = []
    for p in metric_dir.glob("probe_result_*.json"):
        try:
            nums.append(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    if not nums:
        return {"source": "none", "values": []}
    n = max(nums)
    data = json.loads((metric_dir / f"probe_result_{n}.json").read_text())
    # New schema uses `standard_threshold`; legacy probers wrote just
    # `threshold`. Surface both names so the frontend doesn't have to know.
    std_th = data.get("standard_threshold", data.get("threshold"))
    return {
        "source": "completed",
        "run_index": n,
        "metric_name": data.get("metric_name"),
        "threshold": std_th,
        "standard_threshold": std_th,
        "acceptable_threshold": data.get("acceptable_threshold"),
        "direction": data.get("direction"),
        "status": data.get("status"),
        "values": data.get("values", []),
    }


# ── Payload builders ─────────────────────────────────────────────────────────
def _state_payload(state: RunState) -> dict:
    rec = asdict(state.record)
    rec["busy"] = _busy
    # Surface the last background-action failure (if any) so the polling UI can
    # show it — previously these came back as an HTTP 500 on the POST.
    rec["last_error"] = _last_error.get(state.record.run_id)
    return rec


def _stage1_artifact(state: RunState) -> dict:
    p = state.artifact_path(PROBE_CONFIDENCED)
    if not p.exists():
        return {"probe_designs": None}
    return json.loads(p.read_text())


def _stage2_artifact(state: RunState) -> dict:
    p = state.artifact_path(DEV_DOC_CONFIDENCED)
    if not p.exists():
        return {"dev_plans": None}
    return json.loads(p.read_text())


@app.post("/api/cancel")
def cancel():
    """Kill any in-flight stage action and reset the owning run's phase.

    Returns the run that was killed (if any) so the UI can refresh it.
    """
    rid = _current_run_id
    if rid:
        # Mark intentional so _run_action doesn't record this as a `last_error`.
        _cancelled.add(rid)
    killed = cancel_current()
    affected: str | None = None
    if rid:
        try:
            state = load_run(rid)
            new_phase = "ready" if state.record.stage == 4 else "input"
            # If we cancelled in the middle of a fix-plan generation, drop the
            # half-set round pointer — there's no usable JSON to render and
            # the UI shouldn't get stuck on the fix-plan view.
            action = state.record.current_action or ""
            if action.startswith("fix-plan-generate:") or action.startswith("fix-plan-confidence:"):
                state.set_fix_plan_round(None)
            state.set_action(None)
            state.set_phase(state.record.stage, new_phase)
            _last_error.pop(rid, None)
            affected = rid
        except FileNotFoundError:
            pass
    return {"killed": killed, "run": affected}


@app.get("/api/health")
def health():
    return {"ok": True, "busy": _busy}


# Convenience: `python -m server.app` runs uvicorn with reload off.
# API_PORT env var overrides the default 8765 (codex backend uses 8766).
if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("API_PORT", "8765")))
