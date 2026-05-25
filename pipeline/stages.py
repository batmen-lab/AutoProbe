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
from hard_prompt.agent_fix_plan_gen import PROMPT_NINE, PROMPT_TEN
from hard_prompt.agent_fix_plan_apply import PROMPT_ELEVEN
from hard_prompt.auto_research_prompt_patch import (
    PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION,
    PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT,
)

from . import snapshot_git as snap
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

    # Regenerating invalidates any tried-index list (new candidates won't share
    # ordinal positions with the old set).
    state.record.tried_probe_indices = []
    state.record.tried_plan_indices = []
    state.record.probe_index = None
    state.record.plan_index = None
    state.set_phase(Stage.ONE, "running")
    state.set_action("probe-generate")

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

    state.set_action(None)
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
    state.set_action("auto-research-setup")

    agent_call(
        PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION,
        cwd=state.workspace,
        log_path=state.log_path,
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

    # Tag the post-setup train.py as `pre-iter` — the auto-research equivalent
    # of post-stage-3 implementation. revert_to(stage=4) restores from this.
    snap.commit_train(state.workspace, "auto-research setup (pre-iter)", tag="pre-iter")

    # ── Indexing cleanup ──────────────────────────────────────────────────
    # Setup produced two validation runs (post-prober + post-commentor). We
    # seed best_value off the post-commentor tail_mean first, then wipe the
    # setup artifacts so iter 1 produces probe_result_1 / change_log_1.
    seed = _read_tail_mean_and_direction(state.workspace)
    if seed is not None:
        state.record.auto_research_best_value = seed[0]
        state.record.auto_research_best_direction = seed[1]

    metric_dir = state.workspace / ".agent_probe" / "metric"
    if metric_dir.exists():
        for p in metric_dir.glob("probe_result_*.json"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    plot_dir = state.workspace / ".agent_probe" / "plot"
    if plot_dir.exists():
        for p in plot_dir.glob("probe_result_*.pdf"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    base = state.workspace / ".agent_probe"
    for p in base.glob("change_log_*.txt"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    live_file = state.workspace / ".agent_probe" / "live" / "probe_live.json"
    if live_file.exists():
        try:
            live_file.unlink()
        except FileNotFoundError:
            pass

    # No baseline iteration is recorded — iterations starts empty so the
    # first batch round shows up cleanly as "1st run".
    state.record.iterations = []
    state.record.auto_research_target_runs = 0
    state.record.auto_research_runs_completed = 0

    state.set_action(None)
    state.advance_to(Stage.FOUR)


# ── Stage 2: dev plan ────────────────────────────────────────────────────────
def generate_dev_plans(state: RunState) -> dict:
    if state.record.probe_index is None:
        raise ValueError("Stage 2 needs a selected probe. Run select_probe() first.")

    # Regenerating dev plans invalidates the tried-plan list.
    state.record.tried_plan_indices = []
    state.record.plan_index = None
    state.set_phase(Stage.TWO, "running")
    state.set_action("dev-plan-generate")

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

    state.set_action(None)
    state.set_phase(Stage.TWO, "generated")
    return confidenced_plans


def select_plan(state: RunState, index_1based: int) -> None:
    confidenced = json.loads(state.artifact_path(DEV_DOC_CONFIDENCED).read_text())
    n = len(confidenced.get("dev_plans", []))
    if not (1 <= index_1based <= n):
        raise ValueError(f"index out of range: {index_1based} (have {n})")
    state.select_plan(index_1based)
    state.advance_to(Stage.THREE)


# ── Stage 3: implementation ──────────────────────────────────────────────────
def implement(state: RunState) -> None:
    """Agent writes prober.py, integrates train.py, and we run training once."""
    if state.record.plan_index is None:
        raise ValueError("Stage 3 needs a selected plan. Run select_plan() first.")

    state.set_phase(Stage.THREE, "running")
    state.set_action("implementation-apply")

    confidenced = json.loads(state.artifact_path(DEV_DOC_CONFIDENCED).read_text())
    selected = confidenced["dev_plans"][state.record.plan_index - 1]

    prompt = (
        f"{PROMPT_FIVE}\n\n"
        f"Write prober.py and integrate it into train.py.\n\n"
        f"{json.dumps(selected, indent=2)}"
    )
    agent_call(prompt, cwd=state.workspace, log_path=state.log_path)

    # Tag the post-implement train.py as `pre-iter` — the state stage 4 starts
    # from, and what revert_to(stage=4) restores.
    snap.commit_train(state.workspace, "stage 3 implement (pre-iter)", tag="pre-iter")

    # First training run (with auto-fix loop) to produce probe_result_1.
    state.set_action("post-impl-test-run")
    run_training_with_autofix(state)

    # Round 1 is the stage-3 first run — tag the train.py that produced
    # probe_result_1 so future rounds can revert-on-regression back to it.
    snap.commit_train(state.workspace, "round 1 post (stage 3 first run)", tag="round-1-post")

    # Pull stage-3's first metric into the iteration ledger so the UI can show it.
    snapshot = _read_latest_metric(state.workspace)
    if snapshot is not None:
        state.record_iteration(_iteration_record_from_snapshot(snapshot, note="stage 3 first run"))

    state.set_action(None)
    state.set_phase(Stage.THREE, "done")
    state.advance_to(Stage.FOUR)


def _better_strict(new: float, old: float, direction: str) -> bool:
    """Strict improvement in `direction`. Equal counts as NOT better."""
    if direction == "lower_is_better":
        return new < old
    return new > old


def _round_post_tag(n: int) -> str:
    return f"round-{n}-post"


def _maybe_revert_on_regression(state: RunState, round_idx: int) -> str | None:
    """If round `round_idx`'s tail_mean is worse than round `round_idx - 1`'s,
    restore train.py from `round-{N-1}-post` and return a note string. Else
    commit the kept train.py as `round-N-post` and return None.

    Caller is expected to have already run training and have probe_result_N
    on disk. Tags are written via the snapshot git, NOT by the agent. The
    kept/reverted outcome is appended to version_control.json.
    """
    iters = state.record.iterations
    cur_snap = _read_latest_metric(state.workspace)
    cur_tm = (cur_snap or {}).get("tail_mean")
    direction = (cur_snap or {}).get("direction") or "higher_is_better"
    tag_n = _round_post_tag(round_idx)
    prev_tag = _round_post_tag(round_idx - 1)

    # Without a measurable new metric, treat as no-info: revert to be safe.
    if cur_tm is None:
        if snap.tag_exists(state.workspace, prev_tag):
            snap.restore_train(state.workspace, prev_tag)
        snap.commit_train(
            state.workspace, f"round {round_idx} reverted (no metric)", tag=tag_n,
        )
        state.log_round_outcome(round_idx, "reverted", "no metric", restored_from=prev_tag)
        return "reverted (no metric)"
    # No prior round → nothing to compare; just keep.
    prev_tm: float | None = None
    if len(iters) >= 1:
        prev_tm = iters[-1].get("tail_mean")
        if prev_tm is None:
            prev_tm = iters[-1].get("metric_value")
    if prev_tm is None:
        snap.commit_train(
            state.workspace, f"round {round_idx} kept (baseline)", tag=tag_n,
        )
        state.log_round_outcome(round_idx, "kept", "baseline", tail_mean=cur_tm)
        return None
    if _better_strict(cur_tm, prev_tm, direction):
        snap.commit_train(
            state.workspace, f"round {round_idx} kept (improved)", tag=tag_n,
        )
        state.log_round_outcome(
            round_idx, "kept", "improved",
            tail_mean=cur_tm, prev_tail_mean=prev_tm,
        )
        return None
    # Regressed (or unchanged). Roll train.py back to the prior round's tag
    # and commit a no-op so the round-N-post tag still exists for the next
    # round's revert target.
    reverted = False
    if snap.tag_exists(state.workspace, prev_tag):
        snap.restore_train(state.workspace, prev_tag)
        reverted = True
    snap.commit_train(
        state.workspace, f"round {round_idx} reverted (no improvement)", tag=tag_n,
    )
    state.log_round_outcome(
        round_idx, "reverted", "no improvement",
        tail_mean=cur_tm, prev_tail_mean=prev_tm,
        restored_from=prev_tag if reverted else None,
    )
    return "reverted (no improvement)" if reverted else "reverted (no prior snapshot)"


def _iteration_record_from_snapshot(snapshot: dict, *, note: str | None = None, best_value: float | None = None) -> IterationRecord:
    def _as_str(v):
        return str(v) if v is not None else None
    return IterationRecord(
        index=snapshot["index"],
        metric_name=snapshot.get("metric_name"),
        metric_value=snapshot.get("metric_value"),
        threshold=_as_str(snapshot.get("threshold")),
        acceptable_threshold=_as_str(snapshot.get("acceptable_threshold")),
        tail_mean=snapshot.get("tail_mean"),
        direction=snapshot.get("direction"),
        status=snapshot.get("status"),
        acceptable_met=snapshot.get("acceptable_met"),
        note=note,
        best_value=best_value,
    )


# ── Stage 4: iteration ───────────────────────────────────────────────────────
def iterate_once(state: RunState) -> dict:
    """Run a single improvement iteration. Snapshots train.py first."""
    if not (state.workspace / "prober.py").exists():
        raise ValueError("Stage 4 requires prober.py from stage 3.")

    state.set_phase(Stage.FOUR, "running")

    next_idx = _next_round_index(state)

    state.set_action(f"improving-implement:{next_idx}")
    # Auto-research mode uses a different iteration prompt (regression-aware,
    # comment-driven). Normal mode uses the dev-plan-driven iteration prompt.
    iter_prompt = (
        PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT
        if state.record.debug_flags.get("auto_research")
        else PROMPT_SEVEN
    )
    agent_call(iter_prompt, cwd=state.workspace, log_path=state.log_path)

    state.set_action(f"iteration-test-run:{next_idx}")
    run_training_with_autofix(state)

    # Orchestrator-level revert-on-regression: compares this round's tail_mean
    # to the prior iteration record and restores train.py from
    # round-{N-1}-post if worsened. Also tags the resulting train.py as
    # round-{N}-post so future rounds can revert back to it.
    revert_note = _maybe_revert_on_regression(state, next_idx)
    snapshot = _read_latest_metric(state.workspace)
    if snapshot is not None:
        rec = _iteration_record_from_snapshot(snapshot, note=revert_note)
    else:
        rec = IterationRecord(index=next_idx, note=revert_note)
    state.record_iteration(rec)
    state.set_action(None)
    state.set_phase(Stage.FOUR, "ready")
    return {"iteration": rec.__dict__, "passed": rec.status == "PASS"}


# ── Stage 4: fix-plan flow ──────────────────────────────────────────────────
def _next_round_index(state: RunState) -> int:
    """The 1-based index of the next iteration about to run — same number
    used for fix_plans_N.json, round-N-post tags, and probe_result_N.json.
    Source of truth is the count of probe_result_*.json files on disk.
    """
    metric_dir = state.workspace / ".agent_probe" / "metric"
    existing = _glob_indices(metric_dir, "probe_result_*.json")
    return (max(existing) + 1) if existing else 1


# Legacy alias — kept so call sites in the fix-plan flow don't need to change
# while the codebase stabilises. Prefer `_next_round_index` in new code.
_next_fix_plan_round = _next_round_index


def generate_fix_plans(state: RunState, hint: str | None = None) -> dict:
    """Stage 4 fix flow: agent reads repo + history, writes 3 candidate fix
    plans, supervisor agent fills in confidence. Auto-research is independent
    of this path and uses its own iteration prompt instead.

    `hint` is an optional user-supplied string passed verbatim to the
    generator agent. The agent treats it as non-binding direction and the
    supervisor (PROMPT_TEN) scores plans on the merits regardless.
    """
    if state.record.debug_flags.get("auto_research"):
        raise ValueError("Fix-plan flow is not used in auto-research mode.")
    if not (state.workspace / "prober.py").exists():
        raise ValueError("Fix-plan flow requires prober.py from stage 3.")
    if not state.record.iterations:
        raise ValueError("Fix-plan flow requires at least one iteration to learn from.")

    state.set_phase(Stage.FOUR, "running")
    round_idx = _next_fix_plan_round(state)
    fix_plans_dir = state.workspace / ".agent_probe" / "fix_plans"
    fix_plans_dir.mkdir(parents=True, exist_ok=True)
    target_file = fix_plans_dir / f"fix_plans_{round_idx}.json"

    # Clear any stale file from a previously-cancelled generation so we don't
    # render leftover data if the agent fails to write.
    try:
        target_file.unlink()
    except FileNotFoundError:
        pass

    hint_clean = (hint or "").strip()
    hint_block = f"\n\nUser hint:\n{hint_clean}\n" if hint_clean else ""

    state.set_action(f"fix-plan-generate:{round_idx}")
    agent_call(
        f"{PROMPT_NINE}{round_idx}{hint_block}",
        cwd=state.workspace,
        log_path=state.log_path,
    )

    state.set_action(f"fix-plan-confidence:{round_idx}")
    agent_call(
        f"{PROMPT_TEN}{round_idx}",
        cwd=state.workspace,
        log_path=state.log_path,
    )

    if not target_file.exists():
        state.set_action(None)
        state.set_phase(Stage.FOUR, "ready")
        raise RuntimeError(f"Fix-plan generation did not produce {target_file.name}")

    # Mirror the generated plans into the response folder. Chosen index is
    # filled in later (when select_and_apply_fix_plan runs).
    try:
        data = json.loads(target_file.read_text())
        _persist_fix_plans(state, round_idx, data, chosen_index=None)
    except json.JSONDecodeError:
        pass

    state.set_fix_plan_round(round_idx)
    state.set_action(None)
    state.set_phase(Stage.FOUR, "fix-plans-ready")
    return read_fix_plans(state)


def read_fix_plans(state: RunState) -> dict:
    round_idx = state.record.fix_plan_round
    if round_idx is None:
        return {"round": None, "fix_plans": None}
    p = state.workspace / ".agent_probe" / "fix_plans" / f"fix_plans_{round_idx}.json"
    if not p.exists():
        return {"round": round_idx, "fix_plans": None}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"round": round_idx, "fix_plans": None}
    return {"round": round_idx, "fix_plans": data.get("fix_plans")}


def _trd_signed(iters: list) -> float | None:
    """Backend mirror of the frontend's TRD: signed delta between latest
    tail_mean and tail_mean 3 rounds ago, oriented so positive = better.
    Returns None when there aren't enough rows (< 4) to judge.
    """
    if len(iters) < 4:
        return None
    last = iters[-1]
    ref = iters[-4]
    def _tm(r: dict):
        tm = r.get("tail_mean")
        return tm if tm is not None else r.get("metric_value")
    latest = _tm(last)
    refv = _tm(ref)
    if latest is None or refv is None:
        return None
    direction = last.get("direction") or ref.get("direction") or "higher_is_better"
    raw = latest - refv
    return -raw if direction == "lower_is_better" else raw


def _at_terminal_state(state: RunState) -> bool:
    """Auto-fix-loop break condition. Mirrors frontend's classifyFail.

    Terminal cases:
      - PASS (any time).
      - >= 4 rows + improving + AT met → best-effort.
      - >= 4 rows + anything else (FAIL) → stagnant.

    Effectively the auto-loop hard-caps at 3 fix-attempt rounds: it either
    PASSes by then, lands on best-effort, or hands off to the user via the
    stagnant modal.
    """
    iters = state.record.iterations
    if not iters:
        return True
    last = iters[-1]
    if last.get("status") == "PASS":
        return True
    if len(iters) < 4:
        return False  # not enough rows yet — keep iterating
    trd = _trd_signed(iters)
    last_tm = last.get("tail_mean")
    if last_tm is None:
        last_tm = last.get("metric_value") or 1.0
    noise = max(0.01 * abs(last_tm or 1.0), 1e-6)
    improving = trd is not None and trd > noise
    am = last.get("acceptable_met") is True
    if improving and am:
        return True  # best-effort
    return True      # stagnant covers everything else after 4 rows


def _persist_fix_plans(
    state: RunState,
    round_idx: int,
    data: dict,
    chosen_index: int | None,
) -> None:
    """Save the per-round fix-plans JSON into the run's response folder so
    it survives workspace cleanup. Augments with the chosen plan index for
    audit. Mirrors how stage 1/2 store probe_designs / dev_doc.
    """
    payload = dict(data)
    if chosen_index is not None:
        payload["chosen_index"] = chosen_index
    payload["round"] = round_idx
    state.run_dir.mkdir(parents=True, exist_ok=True)
    (state.run_dir / f"fix_plans_{round_idx}.json").write_text(
        json.dumps(payload, indent=2)
    )


def _pick_highest_confidence(plans: list) -> int:
    """Return the 0-based index of the highest-confidence plan; ties → first."""
    best_idx, best_conf = 0, -1.0
    for i, p in enumerate(plans):
        try:
            c = float(p.get("confidence") or 0)
        except (TypeError, ValueError):
            c = 0.0
        if c > best_conf:
            best_conf = c
            best_idx = i
    return best_idx


def _do_auto_fix_round(state: RunState) -> None:
    """One round inside auto_fix_loop: generate plans, auto-pick highest
    confidence, apply, run training, record the iteration. Inlined here
    rather than calling generate_fix_plans + select_and_apply_fix_plan so
    the loop owns phase management end-to-end (no flickers between rounds).
    """
    round_idx = _next_fix_plan_round(state)
    fix_plans_dir = state.workspace / ".agent_probe" / "fix_plans"
    fix_plans_dir.mkdir(parents=True, exist_ok=True)
    target_file = fix_plans_dir / f"fix_plans_{round_idx}.json"
    try:
        target_file.unlink()
    except FileNotFoundError:
        pass

    state.set_action(f"fix-plan-generate:{round_idx}")
    agent_call(
        f"{PROMPT_NINE}{round_idx}",
        cwd=state.workspace,
        log_path=state.log_path,
    )

    state.set_action(f"fix-plan-confidence:{round_idx}")
    agent_call(
        f"{PROMPT_TEN}{round_idx}",
        cwd=state.workspace,
        log_path=state.log_path,
    )

    if not target_file.exists():
        raise RuntimeError(f"Fix-plan generation did not produce {target_file.name}")
    data = json.loads(target_file.read_text())
    plans = data.get("fix_plans") or []
    if not plans:
        raise RuntimeError(f"Fix plans file is empty: {target_file.name}")

    best_idx = _pick_highest_confidence(plans)
    selected = plans[best_idx]
    one_based = best_idx + 1

    state.record.fix_plan_round = round_idx
    state.record.fix_plan_index = one_based
    state.save()

    # Copy the fix-plan JSON into the response folder so it's preserved with
    # the run history (the workspace copy is ephemeral and gets cleaned on
    # revert; the response copy is the durable record).
    _persist_fix_plans(state, round_idx, data, chosen_index=one_based)

    state.set_action(f"fix-plan-apply:{round_idx}")
    agent_call(
        f"{PROMPT_ELEVEN}\n\n{json.dumps(selected, indent=2)}",
        cwd=state.workspace,
        log_path=state.log_path,
    )

    state.set_action(f"fix-plan-test-run:{round_idx}")
    run_training_with_autofix(state)

    # Revert-on-regression at the orchestrator level. If the new tail_mean is
    # worse than the prior round's, restore train.py from round-{N-1}-post.
    # Either way, tag the resulting train.py as round-{N}-post.
    revert_note = _maybe_revert_on_regression(state, round_idx)

    snapshot = _read_latest_metric(state.workspace)
    title = (selected.get("title") or "").strip()
    base_note = f"auto-pick fix plan #{one_based}: {title}"
    note = f"{base_note} — {revert_note}" if revert_note else base_note
    if snapshot is not None:
        rec = _iteration_record_from_snapshot(snapshot, note=note)
    else:
        rec = IterationRecord(index=round_idx, note=f"{base_note} (no metric)")
    rec.fix_plan_chosen_index = one_based
    state.record_iteration(rec)

    # Clear the pointer — auto-pilot never leaves a half-open round.
    state.record.fix_plan_round = None
    state.record.fix_plan_index = None
    state.save()


def auto_fix_loop(state: RunState) -> dict:
    """Stage-4 auto-pilot. Loops fix-plan rounds (auto-pick highest
    confidence) until a terminal state. Terminal mirrors the frontend's
    classifyFail:
      - last status == PASS
      - >= 4 iteration rows + improving + AT met (best-effort)
      - >= 4 iteration rows + not improving (stagnant)
    Auto-research mode is unrelated and uses its own path.
    """
    if state.record.debug_flags.get("auto_research"):
        raise ValueError("Auto-fix-loop is not used in auto-research mode.")
    if not (state.workspace / "prober.py").exists():
        raise ValueError("Stage 4 requires prober.py from stage 3.")
    if not state.record.iterations:
        raise ValueError("Auto-fix-loop needs at least one iteration to learn from.")

    state.set_phase(Stage.FOUR, "running")
    rounds = 0
    try:
        while not _at_terminal_state(state):
            _do_auto_fix_round(state)
            rounds += 1
    finally:
        state.set_action(None)
        state.set_phase(Stage.FOUR, "ready")
    return {"completed": True, "rounds": rounds}


def select_and_apply_fix_plan(state: RunState, index_1based: int) -> dict:
    """User picked one of the 3 candidate fix plans. Apply it + run training,
    then record an iteration as if iterate_once had run.
    """
    if state.record.debug_flags.get("auto_research"):
        raise ValueError("Fix-plan flow is not used in auto-research mode.")
    if state.record.fix_plan_round is None:
        raise ValueError("No open fix-plan round.")
    round_idx = state.record.fix_plan_round
    fp_path = state.workspace / ".agent_probe" / "fix_plans" / f"fix_plans_{round_idx}.json"
    if not fp_path.exists():
        raise ValueError(f"Fix-plan file missing: {fp_path}")
    data = json.loads(fp_path.read_text())
    plans = data.get("fix_plans") or []
    if not (1 <= index_1based <= len(plans)):
        raise ValueError(f"Fix plan index out of range: {index_1based} (have {len(plans)})")
    selected = plans[index_1based - 1]

    state.set_fix_plan_index(index_1based)
    state.set_phase(Stage.FOUR, "running")

    # Persist the per-round fix plans (with chosen index) into the response
    # folder for durable history. Replaces the file-snapshot v_N.py logic —
    # snapshot.git handles train.py revertability.
    _persist_fix_plans(state, round_idx, data, chosen_index=index_1based)

    state.set_action(f"fix-plan-apply:{round_idx}")
    prompt = (
        f"{PROMPT_ELEVEN}\n\n"
        f"{json.dumps(selected, indent=2)}"
    )
    agent_call(prompt, cwd=state.workspace, log_path=state.log_path)

    state.set_action(f"fix-plan-test-run:{round_idx}")
    run_training_with_autofix(state)

    # Orchestrator-level revert-on-regression — same as the auto-pilot path.
    revert_note = _maybe_revert_on_regression(state, round_idx)

    snapshot = _read_latest_metric(state.workspace)
    title = (selected.get("title") or "").strip()
    base_note = f"fix plan #{index_1based}: {title}"
    note = f"{base_note} — {revert_note}" if revert_note else base_note
    if snapshot is not None:
        rec = _iteration_record_from_snapshot(snapshot, note=note)
    else:
        rec = IterationRecord(index=round_idx, note=f"{base_note} (no metric)")
    rec.fix_plan_chosen_index = index_1based
    state.record_iteration(rec)

    # Close the fix-plan round — the file stays on disk for history; the
    # round/index pointer clears so the UI reverts to the ready state.
    state.set_fix_plan_round(None)
    state.set_action(None)
    state.set_phase(Stage.FOUR, "ready")
    return {"iteration": rec.__dict__, "passed": rec.status == "PASS"}


# ── Auto-research batch ──────────────────────────────────────────────────────
def _better(value: float, best: float, direction: str) -> bool:
    if direction == "lower_is_better":
        return value < best
    # default to higher_is_better
    return value > best


def auto_research_iterate_batch(state: RunState, count: int) -> dict:
    """Run `count` auto-research rounds with revert-on-regression.

    Each round: snapshot train.py → run agent → run training → compare new
    `tail_mean` to the running best. If improved, keep the change and update
    best. If not, restore train.py from the pre-iteration snapshot so the
    workspace only ever holds the best version seen. Each recorded
    IterationRecord stores `best_value` so the UI's per-run chart is
    monotonic by construction.
    """
    if not state.record.debug_flags.get("auto_research"):
        raise ValueError("auto_research_iterate_batch requires auto-research mode.")
    if not (state.workspace / "prober.py").exists():
        raise ValueError("Auto-research batch requires prober.py from setup.")
    if count <= 0:
        raise ValueError("count must be >= 1")

    state.set_phase(Stage.FOUR, "running")
    state.record.auto_research_target_runs = count
    state.record.auto_research_runs_completed = 0
    state.save()

    # Seed best from the latest existing probe result if we haven't yet.
    best = state.record.auto_research_best_value
    direction = state.record.auto_research_best_direction
    if best is None:
        seed = _read_tail_mean_and_direction(state.workspace)
        if seed is not None:
            best, direction = seed
            state.record.auto_research_best_value = best
            state.record.auto_research_best_direction = direction
            state.save()

    # The "pre-iteration" snapshot for batch round k is whatever
    # round-{k-1}-post points at (or `pre-iter` for the very first round).
    # We never write a separate pre-snapshot — we just remember the ref to
    # restore from on regression.
    for k in range(1, count + 1):
        next_idx = _next_round_index(state)
        prev_ref = (
            f"round-{next_idx - 1}-post"
            if snap.tag_exists(state.workspace, f"round-{next_idx - 1}-post")
            else "pre-iter"
        )

        state.set_action(f"auto-research-improving:{k}:{count}")
        agent_call(
            PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT,
            cwd=state.workspace,
            log_path=state.log_path,
        )

        state.set_action(f"auto-research-test:{k}:{count}")
        run_training_with_autofix(state)

        tag_n = _round_post_tag(next_idx)

        snapshot = _read_latest_metric(state.workspace)
        if snapshot is None:
            # No metric produced — treat as a no-op round; revert to be safe.
            snap.restore_train(state.workspace, prev_ref)
            snap.commit_train(
                state.workspace, f"round {next_idx} reverted (no metric)", tag=tag_n,
            )
            state.log_round_outcome(
                next_idx, "reverted", "no metric",
                source="auto-research", restored_from=prev_ref,
            )
            state.record.auto_research_runs_completed = k
            state.save()
            continue

        raw = snapshot.get("raw", {}) or {}
        tail_mean = raw.get("tail_mean")
        if tail_mean is None:
            # Fall back to last value if prober didn't emit tail_mean.
            tail_mean = snapshot.get("metric_value")
        cur = float(tail_mean) if tail_mean is not None else None
        direction = raw.get("direction", direction or "higher_is_better")

        if cur is None:
            snap.restore_train(state.workspace, prev_ref)
            snap.commit_train(
                state.workspace, f"round {next_idx} reverted (no metric)", tag=tag_n,
            )
            state.log_round_outcome(
                next_idx, "reverted", "no metric",
                source="auto-research", restored_from=prev_ref,
            )
            note = "reverted (no metric)"
        elif best is None:
            best = cur
            snap.commit_train(
                state.workspace, f"round {next_idx} kept (baseline)", tag=tag_n,
            )
            state.log_round_outcome(
                next_idx, "kept", "baseline of batch",
                source="auto-research", tail_mean=cur,
            )
            note = "kept (baseline of batch)"
        elif _better(cur, best, direction):
            best = cur
            snap.commit_train(
                state.workspace, f"round {next_idx} kept (improved)", tag=tag_n,
            )
            state.log_round_outcome(
                next_idx, "kept", "improved",
                source="auto-research", tail_mean=cur, prev_best=best,
            )
            note = "kept (improved)"
        else:
            # Orchestrator-level revert — train.py rewinds to prev_ref.
            snap.restore_train(state.workspace, prev_ref)
            snap.commit_train(
                state.workspace, f"round {next_idx} reverted (no improvement)", tag=tag_n,
            )
            state.log_round_outcome(
                next_idx, "reverted", "no improvement",
                source="auto-research", tail_mean=cur, prev_best=best,
                restored_from=prev_ref,
            )
            note = "reverted (no improvement)"

        state.record.auto_research_best_value = best
        state.record.auto_research_best_direction = direction
        # Auto-research has no thresholds; build the record so threshold fields
        # stay None even if the snapshot carries them from a stale schema.
        rec = _iteration_record_from_snapshot(snapshot, note=note, best_value=best)
        rec.threshold = None
        rec.acceptable_threshold = None
        rec.acceptable_met = None
        state.record_iteration(rec)
        state.record.auto_research_runs_completed = k
        state.save()

    state.set_action(None)
    state.set_phase(Stage.FOUR, "ready")
    return {
        "completed": count,
        "best_value": state.record.auto_research_best_value,
        "direction": state.record.auto_research_best_direction,
    }


def _read_tail_mean_and_direction(workspace: Path) -> tuple[float, str] | None:
    """Return (tail_mean, direction) from the latest probe_result, if any."""
    metric_dir = workspace / ".agent_probe" / "metric"
    if not metric_dir.exists():
        return None
    nums = _glob_indices(metric_dir, "probe_result_*.json")
    if not nums:
        return None
    n = max(nums)
    data = json.loads((metric_dir / f"probe_result_{n}.json").read_text())
    tm = data.get("tail_mean")
    if tm is None:
        # Fall back to mean of values if tail_mean wasn't written.
        vals = [v.get("value") for v in (data.get("values") or []) if isinstance(v, dict)]
        if not vals:
            return None
        tm = sum(vals[-5:]) / max(1, len(vals[-5:]))
    direction = data.get("direction", "higher_is_better")
    return float(tm), direction


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
    # Prober writes `standard_threshold` in the new schema; legacy probers may
    # still write `threshold` only. Accept both, prefer the new field.
    std_th = data.get("standard_threshold", data.get("threshold"))
    acc_th = data.get("acceptable_threshold")
    return {
        "index": n,
        "metric_name": data.get("metric_name"),
        "metric_value": last_value,
        "threshold": std_th,
        "acceptable_threshold": acc_th,
        "tail_mean": data.get("tail_mean"),
        "direction": data.get("direction"),
        "status": data.get("status"),
        "acceptable_met": data.get("acceptable_met"),
        "raw": data,
    }


