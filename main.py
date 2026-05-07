"""CLI driver for the 4-stage agentic probe pipeline.

The same pipeline that the Next.js frontend drives via the FastAPI server.
Use this for headless / scripted runs.

Usage:
    python main.py --workspace ./mimic
    python main.py --resume <run_id>

Stage 1: ask for context, generate probe designs, prompt user to pick one
Stage 2: generate dev plans, prompt user to pick one
Stage 3: agent writes prober.py, integrates train.py, runs training once
Stage 4: iterate N times (or until probe passes)

Backward navigation is exposed via `--revert <run_id> --to-stage N`.

Debug-flag-gated features (auto-research, threshold override, etc.) are
preserved in the pipeline module but not driven from this CLI by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import (
    Stage,
    list_runs,
    load_run,
    new_run,
)
from pipeline import stages as stages_mod
from pipeline.workspace import open_workspace, list_recent_workspaces


def _ask(text: str) -> str:
    print(f"\n{text}")
    return input(">> ").strip()


def _ask_int(text: str, lo: int, hi: int) -> int:
    while True:
        raw = _ask(text)
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  enter an int in [{lo}, {hi}]")


def _open_workspace_interactive() -> Path:
    s = list_recent_workspaces()
    if s.recent_workspaces:
        print("\nRecent workspaces:")
        for i, w in enumerate(s.recent_workspaces, 1):
            print(f"  [{i}] {w}")
        print("  [0] enter a new path")
        idx = _ask_int("Pick one:", 0, len(s.recent_workspaces))
        if idx > 0:
            return Path(open_workspace(s.recent_workspaces[idx - 1]).current_workspace)

    path = _ask("Enter workspace path (must contain train.py):")
    return Path(open_workspace(path).current_workspace)


def _print_probes(state) -> None:
    p = state.artifact_path("probe_confidenced.json")
    print("\n" + p.read_text())


def _print_plans(state) -> None:
    p = state.artifact_path("dev_doc_confidenced.json")
    print("\n" + p.read_text())


def cmd_run(args) -> None:
    workspace = Path(args.workspace) if args.workspace else _open_workspace_interactive()

    if args.resume:
        state = load_run(args.resume)
    else:
        state = new_run(workspace)
        print(f"[New run] {state.record.run_id} on workspace {state.record.workspace}")

    # Stage 1
    if state.record.stage == Stage.ONE and state.record.phase == "input":
        ctx = state.record.context or _ask("Project context (1–2 sentences + dataset note):")
        state.set_context(ctx)
        print("Generating probes…")
        stages_mod.generate_probes(state)
    if state.record.stage == Stage.ONE and state.record.phase == "generated":
        _print_probes(state)
        idx = _ask_int("Pick probe (1–10):", 1, 10)
        stages_mod.select_probe(state, idx)

    # Stage 2
    if state.record.stage == Stage.TWO and state.record.phase == "input":
        print("Generating dev plans…")
        stages_mod.generate_dev_plans(state)
    if state.record.stage == Stage.TWO and state.record.phase == "generated":
        _print_plans(state)
        idx = _ask_int("Pick plan (1–3):", 1, 3)
        stages_mod.select_plan(state, idx)

    # Stage 3
    if state.record.stage == Stage.THREE and state.record.phase == "input":
        print("Implementing prober + integrating train.py…")
        stages_mod.implement(state)
        last = state.record.iterations[-1] if state.record.iterations else None
        if last:
            print(f"Stage 3 first run: {last.get('metric_name')} = {last.get('metric_value')} ({last.get('status')})")

    # Stage 4
    if state.record.stage == Stage.FOUR:
        n = args.iterations or 3
        for i in range(n):
            if stages_mod.probe_passed(state):
                print("Probe PASSED — stopping early.")
                break
            print(f"Iteration {i + 1}/{n}…")
            result = stages_mod.iterate_once(state)
            print(f"  -> {result['iteration']}")
        print("Done.")


def cmd_revert(args) -> None:
    state = load_run(args.run_id)
    result = state.revert_to(args.to_stage)
    print(f"Reverted to stage {args.to_stage}.")
    for d in result["deleted"]:
        print(f"  deleted: {d}")


def cmd_list(_args) -> None:
    runs = list_runs()
    for r in runs:
        print(f"{r['run_id']}  stage={r['stage']}  phase={r['phase']}  ws={r['workspace']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_run = sub.add_parser("run", help="Run the pipeline (default).")
    p_run.add_argument("--workspace", help="Path to project (must contain train.py)")
    p_run.add_argument("--resume", help="Resume an existing run by id")
    p_run.add_argument("--iterations", type=int, default=3)
    p_run.set_defaults(func=cmd_run)

    p_rev = sub.add_parser("revert", help="Revert a run to an earlier stage.")
    p_rev.add_argument("run_id")
    p_rev.add_argument("--to-stage", type=int, required=True, choices=[1, 2, 3, 4])
    p_rev.set_defaults(func=cmd_revert)

    p_list = sub.add_parser("list", help="List runs.")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if args.cmd is None:
        # Default: interactive run.
        args = parser.parse_args(["run"])
    args.func(args)


if __name__ == "__main__":
    main()
