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
_action_lock = asyncio.Lock()
_current_run_id: str | None = None


async def _run_blocking(fn, *args, **kwargs):
    """Run blocking pipeline fn in a thread. Wraps with the active-stage lock.

    Records which run owns the in-flight action so cancel() can reset its phase.
    """
    global _current_run_id
    if _action_lock.locked():
        raise HTTPException(409, "Another stage action is already running.")
    rid = args[0].record.run_id if args and isinstance(args[0], RunState) else None
    async with _action_lock:
        _current_run_id = rid
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        finally:
            _current_run_id = None


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


class ListDirBody(BaseModel):
    path: str


class ThresholdBody(BaseModel):
    value: str


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
    await _run_blocking(stages_mod.generate_probes, state)
    state = load_run(run_id)  # reload after blocking write
    return _stage1_artifact(state)


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
    await _run_blocking(stages_mod.auto_research_setup, state)
    state = load_run(run_id)
    return _state_payload(state)


# ── Stage 2 ──────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/stage2/generate")
async def stage2_generate(run_id: str):
    state = load_run(run_id)
    await _run_blocking(stages_mod.generate_dev_plans, state)
    state = load_run(run_id)
    return _stage2_artifact(state)


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
    await _run_blocking(stages_mod.implement, state)
    state = load_run(run_id)
    return _state_payload(state)


@app.post("/api/runs/{run_id}/stage3/threshold")
def stage3_threshold(run_id: str, body: ThresholdBody):
    """Override the selected dev plan's threshold before running implement."""
    state = load_run(run_id)
    try:
        stages_mod.override_threshold(state, body.value.strip())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _state_payload(state)


# ── Stage 4 ──────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/stage4/iterate")
async def stage4_iterate(run_id: str):
    state = load_run(run_id)
    await _run_blocking(stages_mod.iterate_once, state)
    state = load_run(run_id)
    return _state_payload(state)


# ── Revert ───────────────────────────────────────────────────────────────────
@app.post("/api/runs/{run_id}/revert")
def revert(run_id: str, body: RevertBody):
    state = load_run(run_id)
    try:
        result = state.revert_to(body.to_stage)
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
    return {
        "source": "completed",
        "run_index": n,
        "metric_name": data.get("metric_name"),
        "threshold": data.get("threshold"),
        "direction": data.get("direction"),
        "status": data.get("status"),
        "values": data.get("values", []),
    }


# ── Payload builders ─────────────────────────────────────────────────────────
def _state_payload(state: RunState) -> dict:
    rec = asdict(state.record)
    rec["busy"] = _action_lock.locked()
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
    killed = cancel_current()
    affected: str | None = None
    if rid:
        try:
            state = load_run(rid)
            new_phase = "ready" if state.record.stage == 4 else "input"
            state.set_phase(state.record.stage, new_phase)
            affected = rid
        except FileNotFoundError:
            pass
    return {"killed": killed, "run": affected}


@app.get("/api/health")
def health():
    return {"ok": True, "busy": _action_lock.locked()}


# Convenience: `python -m server.app` runs uvicorn with reload off.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
