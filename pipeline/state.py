"""RunState — single source of truth for one run's progress + artifacts.

A run is rooted at response/<run_id>/ and references one workspace directory
(the project to instrument). The two are tied together for the run's lifetime.

State is captured in stage.json:
    {
      "run_id": "...",
      "workspace": "/abs/path/to/project",
      "created_at": "...",
      "stage": 1,                # current active stage (1..4)
      "phase": "input"|"generated"|"selected"|"running"|"done",
      "context": "...",          # stage 1 input (user's project description)
      "probe_index": 3,          # stage 1 selection (1-based)
      "plan_index": 1,           # stage 2 selection (1-based)
      "iterations": [            # stage 4 history
        {"index": 1, "metric_value": 0.85, "status": "FAIL", ...}
      ],
      "debug_flags": {"auto_research": false, "threshold_override": null}
    }

Backward navigation = `revert_to(stage)`. The target stage's outputs and any
later stages' artifacts are wiped (workspace + run_dir). Stage inputs (e.g.,
context for stage 1) are preserved so the user can edit and re-run.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

from .workspace import RUN_BASE


class Stage(IntEnum):
    ONE = 1   # probe design (NLP)
    TWO = 2   # dev plan (NLP)
    THREE = 3 # implementation (agent)
    FOUR = 4  # iteration (agent)


# ── filenames inside response/<run_id>/ ──────────────────────────────────────
PROBE_DESIGNS = "probe_designs.json"
PROBE_CONFIDENCED = "probe_confidenced.json"
DEV_DOC = "dev_doc.json"
DEV_DOC_CONFIDENCED = "dev_doc_confidenced.json"
STAGE_FILE = "stage.json"
LOG_FILE = "agent.log"


# ── workspace paths ──────────────────────────────────────────────────────────
def _ws_paths(workspace: Path) -> dict[str, Path]:
    base = workspace / ".agent_probe"
    return {
        "agent_probe": base,
        "snapshot": base / "snapshot",
        "metric": base / "metric",
        "plot": base / "plot",
        "live": base / "live",
        "train": workspace / "train.py",
        "prober": workspace / "prober.py",
    }


@dataclass
class IterationRecord:
    index: int
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: str | None = None
    status: str | None = None  # "PASS"/"FAIL"
    note: str | None = None


@dataclass
class StageRecord:
    run_id: str
    workspace: str
    created_at: str
    stage: int = 1
    phase: str = "input"  # input|generated|selected|running|done
    context: str | None = None
    probe_index: int | None = None
    plan_index: int | None = None
    iterations: list[dict] = field(default_factory=list)
    debug_flags: dict = field(default_factory=lambda: {
        "auto_research": False,
        "threshold_override": None,
    })


# ── helpers ──────────────────────────────────────────────────────────────────
def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _write_json(p: Path, data: Any) -> None:
    p.write_text(json.dumps(data, indent=2))


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def _purge_glob(directory: Path, pattern: str, keep_max_index: int = 0) -> None:
    """Delete files matching directory/pattern whose trailing _<n> > keep_max_index.

    Pattern is shell glob; the file's stem must end in `_<int>`. If keep_max_index
    is 0, all matching files are deleted.
    """
    if not directory.exists():
        return
    for p in directory.glob(pattern):
        try:
            n = int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if n > keep_max_index:
            _safe_unlink(p)


# ── RunState ─────────────────────────────────────────────────────────────────
class RunState:
    """One run's filesystem-backed state. Cheap to construct, all I/O is explicit."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.stage_path = run_dir / STAGE_FILE
        self.log_path = run_dir / LOG_FILE
        self._record = self._load()

    @property
    def record(self) -> StageRecord:
        return self._record

    @property
    def workspace(self) -> Path:
        return Path(self._record.workspace)

    def _load(self) -> StageRecord:
        if not self.stage_path.exists():
            raise FileNotFoundError(f"Stage file missing: {self.stage_path}")
        data = _read_json(self.stage_path)
        return StageRecord(**data)

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.stage_path, asdict(self._record))

    # ── stage transitions ────────────────────────────────────────────────────
    def set_phase(self, stage: Stage | int, phase: str) -> None:
        self._record.stage = int(stage)
        self._record.phase = phase
        self.save()

    def advance_to(self, stage: Stage | int) -> None:
        self._record.stage = int(stage)
        self._record.phase = "input"
        self.save()

    # ── stage 1 ──────────────────────────────────────────────────────────────
    def set_context(self, context: str) -> None:
        self._record.context = context
        self.save()

    def select_probe(self, index_1based: int) -> None:
        self._record.probe_index = index_1based
        self.save()

    # ── stage 2 ──────────────────────────────────────────────────────────────
    def select_plan(self, index_1based: int) -> None:
        self._record.plan_index = index_1based
        self.save()

    # ── stage 4 ──────────────────────────────────────────────────────────────
    def record_iteration(self, rec: IterationRecord) -> None:
        self._record.iterations.append(asdict(rec))
        self.save()

    # ── artifact paths ───────────────────────────────────────────────────────
    def artifact_path(self, name: str) -> Path:
        return self.run_dir / name

    # ── backward navigation ──────────────────────────────────────────────────
    def revert_to(self, target_stage: Stage | int) -> dict:
        """Wipe target stage's outputs and any later stages' artifacts.

        Returns a summary of what was deleted (for logging / UI feedback).
        Stage inputs (e.g. context for stage 1, probe selection for stage 2 if
        reverting to stage 3) are preserved.
        """
        target = int(target_stage)
        if target not in (1, 2, 3, 4):
            raise ValueError(f"Invalid target stage: {target}")

        deleted: list[str] = []
        ws = self.workspace
        wp = _ws_paths(ws)
        snapshot_v0 = wp["snapshot"] / "train_version_0.py"
        snapshot_v1 = wp["snapshot"] / "train_version_1.py"

        # Reverting to stage 1: clear stage-1 outputs and everything after.
        if target <= 1:
            for name in (PROBE_DESIGNS, PROBE_CONFIDENCED):
                p = self.run_dir / name
                if p.exists():
                    _safe_unlink(p); deleted.append(str(p))
            self._record.probe_index = None

        # Reverting to stage 2: clear stage-2 outputs and everything after.
        if target <= 2:
            for name in (DEV_DOC, DEV_DOC_CONFIDENCED):
                p = self.run_dir / name
                if p.exists():
                    _safe_unlink(p); deleted.append(str(p))
            self._record.plan_index = None

        # Reverting to stage 3: wipe ALL workspace artifacts; restore baseline.
        if target <= 3:
            if wp["prober"].exists():
                _safe_unlink(wp["prober"]); deleted.append(str(wp["prober"]))
            if snapshot_v0.exists():
                wp["train"].write_text(snapshot_v0.read_text())
                deleted.append(f"{wp['train']} (restored from v0)")
            # delete all metrics / plots / change_logs / version snapshots > 0
            _purge_glob(wp["metric"], "probe_result_*.json", keep_max_index=0)
            _purge_glob(wp["plot"], "probe_result_*.pdf", keep_max_index=0)
            _purge_glob(wp["agent_probe"], "change_log_*.txt", keep_max_index=0)
            _purge_glob(wp["snapshot"], "train_version_*.py", keep_max_index=0)
            _safe_unlink(wp["live"] / "probe_live.json")
            self._record.iterations = []
            # Auto-research mode flag is per-stage-1 choice; reset it.
            self._record.debug_flags["auto_research"] = False
            self._record.debug_flags["threshold_override"] = None

        # Reverting to stage 4: keep stage-3's artifacts (v1 snapshot, prober,
        # probe_result_1, change_log_1) — clear iteration outputs only.
        elif target == 4:
            if snapshot_v1.exists():
                wp["train"].write_text(snapshot_v1.read_text())
                deleted.append(f"{wp['train']} (restored from v1)")
            _purge_glob(wp["metric"], "probe_result_*.json", keep_max_index=1)
            _purge_glob(wp["plot"], "probe_result_*.pdf", keep_max_index=1)
            _purge_glob(wp["agent_probe"], "change_log_*.txt", keep_max_index=1)
            _purge_glob(wp["snapshot"], "train_version_*.py", keep_max_index=1)
            # Drop the live file too — its trajectory is from a now-discarded iter.
            # Fallback in /api/live-metric will read probe_result_1.json.
            _safe_unlink(wp["live"] / "probe_live.json")
            # Keep only the first iteration record (stage-3's first run).
            self._record.iterations = self._record.iterations[:1]

        self._record.stage = target
        self._record.phase = "input" if target in (1, 2) else "ready"
        self.save()
        return {"deleted": deleted, "stage": target, "phase": self._record.phase}


# ── factory ──────────────────────────────────────────────────────────────────
def list_runs(workspace: Path | None = None) -> list[dict]:
    """List runs, optionally filtered to a workspace."""
    RUN_BASE.mkdir(exist_ok=True)
    runs = []
    for p in sorted(RUN_BASE.iterdir(), reverse=True):
        if not p.is_dir() or not p.name.isdigit():
            continue
        sf = p / STAGE_FILE
        if not sf.exists():
            continue
        try:
            data = _read_json(sf)
        except Exception:
            continue
        if workspace is not None and data.get("workspace") != str(workspace):
            continue
        runs.append({
            "run_id": data.get("run_id", p.name),
            "workspace": data.get("workspace"),
            "created_at": data.get("created_at"),
            "stage": data.get("stage", 1),
            "phase": data.get("phase", "input"),
        })
    return runs


def load_run(run_id: str) -> RunState:
    run_dir = RUN_BASE / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return RunState(run_dir)


def new_run(workspace: Path) -> RunState:
    """Create a new run. Snapshots train.py as train_version_0.py."""
    workspace = workspace.resolve()
    if not (workspace / "train.py").exists():
        raise ValueError(f"Workspace missing train.py: {workspace}")

    RUN_BASE.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = RUN_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    record = StageRecord(
        run_id=run_id,
        workspace=str(workspace),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    state_file = run_dir / STAGE_FILE
    _write_json(state_file, asdict(record))

    # Baseline snapshot of train.py for revert-to-stage-3.
    wp = _ws_paths(workspace)
    wp["snapshot"].mkdir(parents=True, exist_ok=True)
    snap_v0 = wp["snapshot"] / "train_version_0.py"
    snap_v0.write_text(wp["train"].read_text())

    return RunState(run_dir)
