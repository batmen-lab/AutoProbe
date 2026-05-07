"""Stage actions — each function does one discrete step.

Stage 1 (NLP):  generate_probes, select_probe
Stage 2 (NLP):  generate_dev_plans, select_plan
Stage 3 (agent): implement
Stage 4 (agent): iterate_once

Each function reads/writes via the RunState passed in and returns the new
phase/output suitable for the API response.
"""

from __future__ import annotations

import json
from pathlib import Path

from hard_prompt.nlp_prober_gen import PROMPT_ONE
from hard_prompt.nlp_prober_confi_comput import PROMPT_TWO
from hard_prompt.nlp_dev_doc_gen import PROMPT_THREE
from hard_prompt.nlp_dd_confi_comput import PROMPT_FOUR
from hard_prompt.agent_dd_implement import PROMPT_FIVE
from hard_prompt.agent_improve_commentor import PROMPT_SIX
from hard_prompt.agent_iterat_improver import PROMPT_SEVEN
from hard_prompt.agent_exception_catcher import PROMPT_EIGHT
from hard_prompt.auto_research_prompt_patch import (
    PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION,
    PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT,
)

from .llm import nlp_call, agent_call
from .state import (
    RunState,
    Stage,
    PROBE_DESIGNS,
    PROBE_CONFIDENCED,
    DEV_DOC,
    DEV_DOC_CONFIDENCED,
    IterationRecord,
)


MAX_FIX_RETRIES = 5


# ── Stage 1: probe design ────────────────────────────────────────────────────
def generate_probes(state: RunState) -> dict:
    """Stage 1: NLP generates probe designs, then a second pass adds confidence."""
    if not state.record.context:
        raise ValueError("Stage 1 needs a context string. Set it via set_context().")

    state.set_phase(Stage.ONE, "running")

    designs = nlp_call(
        f"{PROMPT_ONE}\n\n{state.record.context}",
        log_path=state.log_path,
        label="stage 1.a probe-design generation",
    )
    state.run_dir.joinpath(PROBE_DESIGNS).write_text(json.dumps(designs, indent=2))

    confidenced = nlp_call(
        f"{PROMPT_TWO}\n\n{json.dumps(designs)}",
        log_path=state.log_path,
        label="stage 1.b probe-design confidence",
    )
    state.run_dir.joinpath(PROBE_CONFIDENCED).write_text(json.dumps(confidenced, indent=2))

    state.set_phase(Stage.ONE, "generated")
    return confidenced


def select_probe(state: RunState, index_1based: int) -> None:
    """Stage 1 selection. Advances to stage 2 in 'input' phase."""
    confidenced = json.loads(state.artifact_path(PROBE_CONFIDENCED).read_text())
    n = len(confidenced.get("probe_designs", []))
    if not (1 <= index_1based <= n):
        raise ValueError(f"index out of range: {index_1based} (have {n})")
    state.select_probe(index_1based)
    state.advance_to(Stage.TWO)


def auto_research_setup(state: RunState) -> None:
    """Stage 1 alternative: skip NLP probe/dev-plan generation entirely.

    The agent picks a standard performance metric, writes prober.py + integrates
    train.py, then we run training, run the commentor (which seeds 10
    `# potential_improvement_N:` markers in train.py), then training again to
    validate. This jumps straight from stage 1 to stage 4.
    """
    state.set_phase(Stage.ONE, "running")
    state.record.debug_flags["auto_research"] = True
    state.save()

    agent_call(
        PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION,
        cwd=state.workspace,
        log_path=state.log_path,
    )

    snapshot_dir = state.workspace / ".agent_probe" / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "train_version_1.py").write_text(
        (state.workspace / "train.py").read_text()
    )

    # Validate the integration with a real training run.
    run_training_with_autofix(state)

    # Commentor seeds train.py with the 10 improvement markers used by
    # PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT in stage 4.
    agent_call(
        f"{PROMPT_SIX}\n\nTarget file: train.py",
        cwd=state.workspace,
        log_path=state.log_path,
    )
    # Re-validate after the commentor edits.
    run_training_with_autofix(state)

    snapshot = _read_latest_metric(state.workspace)
    if snapshot is not None:
        state.record_iteration(IterationRecord(
            index=snapshot["index"],
            metric_name=snapshot.get("metric_name"),
            metric_value=snapshot.get("metric_value"),
            threshold=str(snapshot.get("threshold")) if snapshot.get("threshold") is not None else None,
            status=snapshot.get("status"),
            note="auto-research first run",
        ))

    state.advance_to(Stage.FOUR)


# ── Stage 2: dev plan ────────────────────────────────────────────────────────
def generate_dev_plans(state: RunState) -> dict:
    if state.record.probe_index is None:
        raise ValueError("Stage 2 needs a selected probe. Run select_probe() first.")

    state.set_phase(Stage.TWO, "running")

    confidenced = json.loads(state.artifact_path(PROBE_CONFIDENCED).read_text())
    selected = confidenced["probe_designs"][state.record.probe_index - 1]

    plans = nlp_call(
        f"{PROMPT_THREE}\n\n{json.dumps(selected, indent=2)}",
        log_path=state.log_path,
        label="stage 2.a dev-plan generation",
    )
    state.run_dir.joinpath(DEV_DOC).write_text(json.dumps(plans, indent=2))

    confidenced_plans = nlp_call(
        f"{PROMPT_FOUR}\n\n{json.dumps(plans)}",
        log_path=state.log_path,
        label="stage 2.b dev-plan confidence",
    )
    state.run_dir.joinpath(DEV_DOC_CONFIDENCED).write_text(json.dumps(confidenced_plans, indent=2))

    state.set_phase(Stage.TWO, "generated")
    return confidenced_plans


def select_plan(state: RunState, index_1based: int) -> None:
    confidenced = json.loads(state.artifact_path(DEV_DOC_CONFIDENCED).read_text())
    n = len(confidenced.get("dev_plans", []))
    if not (1 <= index_1based <= n):
        raise ValueError(f"index out of range: {index_1based} (have {n})")
    state.select_plan(index_1based)
    state.advance_to(Stage.THREE)


def override_threshold(state: RunState, new_threshold: str) -> None:
    """Replace the selected dev plan's threshold before stage-3 implementation.

    Only valid pre-implement (no prober.py yet). Post-implement threshold changes
    would require also rewriting prober.py + re-evaluating existing metrics —
    deliberately not supported here to keep the surface small.
    """
    if state.record.plan_index is None:
        raise ValueError("Cannot override threshold: no plan selected.")
    if (state.workspace / "prober.py").exists():
        raise ValueError("Cannot override threshold once prober.py exists. Revert to stage 3 first.")
    path = state.artifact_path(DEV_DOC_CONFIDENCED)
    data = json.loads(path.read_text())
    idx = state.record.plan_index - 1
    data["dev_plans"][idx]["threshold"] = new_threshold
    path.write_text(json.dumps(data, indent=2))
    state.record.debug_flags["threshold_override"] = new_threshold
    state.save()


# ── Stage 3: implementation ──────────────────────────────────────────────────
def implement(state: RunState) -> None:
    """Agent writes prober.py, integrates train.py, and we run training once."""
    if state.record.plan_index is None:
        raise ValueError("Stage 3 needs a selected plan. Run select_plan() first.")

    state.set_phase(Stage.THREE, "running")

    confidenced = json.loads(state.artifact_path(DEV_DOC_CONFIDENCED).read_text())
    selected = confidenced["dev_plans"][state.record.plan_index - 1]

    prompt = (
        f"{PROMPT_FIVE}\n\n"
        f"Write prober.py and integrate it into train.py.\n\n"
        f"{json.dumps(selected, indent=2)}"
    )
    agent_call(prompt, cwd=state.workspace, log_path=state.log_path)

    # Snapshot post-stage-3 train.py as train_version_1.py — this is the baseline
    # for stage 4 revert. (Iteration N then snapshots BEFORE running iter N+1.)
    snapshot_dir = state.workspace / ".agent_probe" / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "train_version_1.py").write_text(
        (state.workspace / "train.py").read_text()
    )

    # First training run (with auto-fix loop) to produce probe_result_1.
    run_training_with_autofix(state)

    # Pull stage-3's first metric into the iteration ledger so the UI can show it.
    snapshot = _read_latest_metric(state.workspace)
    if snapshot is not None:
        state.record_iteration(IterationRecord(
            index=snapshot["index"],
            metric_name=snapshot.get("metric_name"),
            metric_value=snapshot.get("metric_value"),
            threshold=str(snapshot.get("threshold")) if snapshot.get("threshold") is not None else None,
            status=snapshot.get("status"),
            note="stage 3 first run",
        ))

    state.set_phase(Stage.THREE, "done")
    state.advance_to(Stage.FOUR)


# ── Stage 4: iteration ───────────────────────────────────────────────────────
def iterate_once(state: RunState) -> dict:
    """Run a single improvement iteration. Snapshots train.py first."""
    if not (state.workspace / "prober.py").exists():
        raise ValueError("Stage 4 requires prober.py from stage 3.")

    state.set_phase(Stage.FOUR, "running")

    snapshot_dir = state.workspace / ".agent_probe" / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Index of the version we're ABOUT to produce.
    next_idx = _next_version_index(snapshot_dir)
    # Snapshot current train.py as train_version_<next_idx>.py BEFORE the agent edits.
    (snapshot_dir / f"train_version_{next_idx}.py").write_text(
        (state.workspace / "train.py").read_text()
    )

    # Auto-research mode uses a different iteration prompt (regression-aware,
    # comment-driven). Normal mode uses the dev-plan-driven iteration prompt.
    iter_prompt = (
        PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT
        if state.record.debug_flags.get("auto_research")
        else PROMPT_SEVEN
    )
    agent_call(iter_prompt, cwd=state.workspace, log_path=state.log_path)
    run_training_with_autofix(state)

    snapshot = _read_latest_metric(state.workspace)
    rec = IterationRecord(
        index=snapshot["index"] if snapshot else next_idx,
        metric_name=snapshot.get("metric_name") if snapshot else None,
        metric_value=snapshot.get("metric_value") if snapshot else None,
        threshold=str(snapshot.get("threshold")) if snapshot and snapshot.get("threshold") is not None else None,
        status=snapshot.get("status") if snapshot else None,
        note=None,
    )
    state.record_iteration(rec)
    state.set_phase(Stage.FOUR, "ready")
    return {"iteration": rec.__dict__, "passed": rec.status == "PASS"}


def probe_passed(state: RunState) -> bool:
    snap = _read_latest_metric(state.workspace)
    return bool(snap and snap.get("status") == "PASS")


# ── exception-catching training loop ─────────────────────────────────────────
import shutil
import subprocess


def run_training_with_autofix(state: RunState) -> None:
    """Run python train.py; on failure, ask agent to fix; retry up to MAX_FIX_RETRIES."""
    metric_dir = state.workspace / ".agent_probe" / "metric"
    plot_dir = state.workspace / ".agent_probe" / "plot"
    existing = _glob_indices(metric_dir, "probe_result_*.json")

    success, err = _run_training(state)
    retries = 0
    while not success:
        if retries >= MAX_FIX_RETRIES:
            raise RuntimeError(f"Could not fix after {MAX_FIX_RETRIES} attempts.\n{err}")
        retries += 1
        _purge_new_artifacts(metric_dir, plot_dir, existing)
        agent_call(f"{PROMPT_EIGHT}\n\nError output:\n{err}", cwd=state.workspace, log_path=state.log_path)
        success, err = _run_training(state)


def _run_training(state: RunState) -> tuple[bool, str]:
    log_path = state.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer `python` if it exists (user may have set it up), otherwise fall back to
    # `python3` — the one we know is on the box. Don't use sys.executable: that's
    # the venv interpreter, which doesn't have the user's training-stack packages.
    interpreter = shutil.which("python") or shutil.which("python3") or "python3"
    with log_path.open("ab") as f:
        f.write(f"\n--- {interpreter} train.py ---\n".encode())
        f.flush()
        result = subprocess.run(
            [interpreter, "train.py"],
            cwd=str(state.workspace),
            capture_output=True,
            text=True,
        )
        f.write(result.stdout.encode())
        f.write(result.stderr.encode())
    return result.returncode == 0, result.stderr


def _glob_indices(directory: Path, pattern: str) -> set[int]:
    out: set[int] = set()
    if not directory.exists():
        return out
    for p in directory.glob(pattern):
        try:
            out.add(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return out


def _purge_new_artifacts(metric_dir: Path, plot_dir: Path, existing: set[int]) -> None:
    for p in metric_dir.glob("probe_result_*.json"):
        try:
            if int(p.stem.rsplit("_", 1)[-1]) not in existing:
                p.unlink(missing_ok=True)
        except ValueError:
            continue
    for p in plot_dir.glob("probe_result_*.pdf"):
        try:
            if int(p.stem.rsplit("_", 1)[-1]) not in existing:
                p.unlink(missing_ok=True)
        except ValueError:
            continue
    # Also clean the live file so a partially-written trajectory from the failed
    # run doesn't bleed into the retry attempt's chart.
    live = metric_dir.parent / "live" / "probe_live.json"
    if live.exists():
        live.unlink(missing_ok=True)


def _read_latest_metric(workspace: Path) -> dict | None:
    metric_dir = workspace / ".agent_probe" / "metric"
    if not metric_dir.exists():
        return None
    nums = _glob_indices(metric_dir, "probe_result_*.json")
    if not nums:
        return None
    n = max(nums)
    data = json.loads((metric_dir / f"probe_result_{n}.json").read_text())
    values = data.get("values") or []
    last_value: float | None = None
    if values:
        v = values[-1]
        last_value = v.get("value") if isinstance(v, dict) else v
    return {
        "index": n,
        "metric_name": data.get("metric_name"),
        "metric_value": last_value,
        "threshold": data.get("threshold"),
        "status": data.get("status"),
        "raw": data,
    }


def _next_version_index(snapshot_dir: Path) -> int:
    nums = _glob_indices(snapshot_dir, "train_version_*.py")
    return (max(nums) + 1) if nums else 1
