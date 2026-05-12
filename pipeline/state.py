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
    # Auto-research only: the running best metric AFTER this iteration
    # (after the orchestrator's revert-on-regression check). This is the
    # value plotted on the monotonic per-run chart. Stays None for normal
    # threshold-gated runs.
    best_value: float | None = None


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
    # Indices already tried-and-reverted, displayed greyed out so the user
    # doesn't reselect them. Cleared when the corresponding candidate set is
    # regenerated (indices would be stale).
    tried_probe_indices: list[int] = field(default_factory=list)
    tried_plan_indices: list[int] = field(default_factory=list)
    # Set by stage actions while in flight; surfaced in the UI status bar so
    # the user knows what's happening during long-running calls. None when idle.
    current_action: str | None = None
    # Auto-research batch state. target_runs is the size of the most recent
    # batch the user kicked off; runs_completed counts what's actually
    # finished. best_value/direction track the running monotonic best so the
    # orchestrator can revert regressions and the UI can plot the chart.
    auto_research_target_runs: int = 0
    auto_research_runs_completed: int = 0
    auto_research_best_value: float | None = None
    auto_research_best_direction: str | None = None


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

    def set_action(self, action: str | None) -> None:
        """Surface what the pipeline is currently doing for the UI status bar."""
        self._record.current_action = action
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
    def revert_to(self, target_stage: Stage | int, keep_workspace: bool = False) -> dict:
        """Land at the END of the target stage (selection cleared, candidates kept).

        Reverting to stage 1 or 2 is "I want to re-pick from the same set of
        candidates": probe_designs / dev_plans are kept, only the index is
        cleared, and the run lands at phase="generated" so the UI shows the
        candidate list ready for re-selection. The just-cleared index is
        appended to `tried_probe_indices` / `tried_plan_indices` so the UI can
        grey out already-tried options. Stage-1's own "Regenerate" button is
        the way to discard candidates and start fresh.

        Reverting across stage 1 (e.g. 3→1) discards stage-2 artifacts because
        they were generated for the old probe and would be stale. Workspace
        artifacts (prober.py, snapshots, metrics, plots, iterations) from
        stage 3+ are wiped whenever the target is ≤ 3.

        Reverting to stage 3 or 4 keeps the existing semantics.

        With `keep_workspace=True` (only valid for target==1): used by the
        "keep changes" path after a stage-4 PASS. We KEEP the modified
        train.py (re-snapshotting it as the new train_version_0 baseline) but
        otherwise reset everything — probe_designs, dev_doc, prober, metrics,
        iterations, tried lists. The run lands at stage=1 phase="input" so
        the user must regenerate probe candidates against the new train.py.
        """
        target = int(target_stage)
        if target not in (1, 2, 3, 4):
            raise ValueError(f"Invalid target stage: {target}")
        if keep_workspace and target != 1:
            raise ValueError("keep_workspace is only supported when reverting to stage 1.")

        deleted: list[str] = []
        ws = self.workspace
        wp = _ws_paths(ws)
        snapshot_v0 = wp["snapshot"] / "train_version_0.py"
        snapshot_v1 = wp["snapshot"] / "train_version_1.py"

        # ── "keep changes" path (post-PASS, stage 4 → stage 1 with new baseline)
        if keep_workspace:
            # Re-baseline train.py: the modified file becomes the new v0.
            wp["snapshot"].mkdir(parents=True, exist_ok=True)
            snapshot_v0.write_text(wp["train"].read_text())
            deleted.append(f"{snapshot_v0} (re-baselined to current train.py)")
            # Wipe prober + all probe/plan/iteration artifacts (probe candidates
            # included — they'll be regenerated against the new train.py).
            if wp["prober"].exists():
                _safe_unlink(wp["prober"]); deleted.append(str(wp["prober"]))
            for name in (PROBE_DESIGNS, PROBE_CONFIDENCED, DEV_DOC, DEV_DOC_CONFIDENCED):
                p = self.run_dir / name
                if p.exists():
                    _safe_unlink(p); deleted.append(str(p))
            _purge_glob(wp["metric"], "probe_result_*.json", keep_max_index=0)
            _purge_glob(wp["plot"], "probe_result_*.pdf", keep_max_index=0)
            _purge_glob(wp["agent_probe"], "change_log_*.txt", keep_max_index=0)
            # Don't blow away the new v0 we just wrote.
            for p in wp["snapshot"].glob("train_version_*.py"):
                try:
                    n = int(p.stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                if n > 0:
                    _safe_unlink(p)
            _safe_unlink(wp["live"] / "probe_live.json")
            self._record.probe_index = None
            self._record.plan_index = None
            self._record.iterations = []
            self._record.tried_probe_indices = []
            self._record.tried_plan_indices = []
            self._record.debug_flags["auto_research"] = False
            self._record.debug_flags["threshold_override"] = None
            self._record.auto_research_target_runs = 0
            self._record.auto_research_runs_completed = 0
            self._record.auto_research_best_value = None
            self._record.auto_research_best_direction = None
            self._record.stage = 1
            self._record.phase = "input"
            self._record.current_action = None
            self.save()
            return {"deleted": deleted, "stage": 1, "phase": "input", "keep_workspace": True}

        # Mark the just-cleared probe as "already tried" so the UI can grey it
        # out — but ONLY when we're fully rolling back from stage 4 to stage 1.
        # Casual back-navigation (2→1, 3→2, 3→1, 4→2, 4→3) is treated as the
        # user changing their mind mid-flight; the selection stays available so
        # they can re-pick it. The "fully turn back" semantic (4→1) is the
        # signal that the probe has been exhausted (whether it passed or not).
        came_from = int(self._record.stage)
        if (
            target == 1
            and came_from == 4
            and self._record.probe_index is not None
            and self._record.probe_index not in self._record.tried_probe_indices
        ):
            self._record.tried_probe_indices.append(self._record.probe_index)

        # Reverting to stage 1: clear the probe SELECTION but keep the
        # generated probe candidates so the user can pick a different one.
        if target <= 1:
            self._record.probe_index = None

        # Reverting *past* stage 1 (i.e. landing at stage 1) discards stage-2
        # dev plans — they were tied to the old probe. Reverting *to* stage 2
        # keeps them so the user can re-pick a plan for the same probe.
        if target <= 1:
            for name in (DEV_DOC, DEV_DOC_CONFIDENCED):
                p = self.run_dir / name
                if p.exists():
                    _safe_unlink(p); deleted.append(str(p))
            self._record.plan_index = None
            # Plan indices were keyed to the just-discarded plan set; stale.
            self._record.tried_plan_indices = []
        elif target == 2:
            self._record.plan_index = None

        # Reverting to stage 3 (or below): wipe ALL workspace artifacts;
        # restore baseline train.py.
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
        # Stages 1/2 land at their end (candidates kept, awaiting re-selection)
        # ONLY when the candidate artifact actually exists. If we wiped past
        # it (e.g. auto-research never produced probe_designs, or the user
        # reverted across stage 1), we drop to "input" so the UI shows the
        # form rather than an empty list.
        if target == 1:
            self._record.phase = (
                "generated"
                if (self.run_dir / PROBE_CONFIDENCED).exists()
                else "input"
            )
        elif target == 2:
            self._record.phase = (
                "generated"
                if (self.run_dir / DEV_DOC_CONFIDENCED).exists()
                else "input"
            )
        else:
            self._record.phase = "ready"
        # Auto-research batch state always resets when reverting past stage 4.
        if target < 4:
            self._record.auto_research_target_runs = 0
            self._record.auto_research_runs_completed = 0
            self._record.auto_research_best_value = None
            self._record.auto_research_best_direction = None
        self._record.current_action = None
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
