"""Workspace ('open folder') management — VS Code-style.

A workspace is a directory containing at minimum train.py. The app remembers
recent workspaces and the currently-opened one in response/_app_state.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
RUN_BASE: Final = REPO_ROOT / "response"
APP_STATE_PATH: Final = RUN_BASE / "_app_state.json"
MAX_RECENT: Final = 10


@dataclass
class AppState:
    current_workspace: str | None
    recent_workspaces: list[str]


def _load() -> AppState:
    if not APP_STATE_PATH.exists():
        return AppState(current_workspace=None, recent_workspaces=[])
    data = json.loads(APP_STATE_PATH.read_text())
    return AppState(
        current_workspace=data.get("current_workspace"),
        recent_workspaces=data.get("recent_workspaces", []),
    )


def _save(state: AppState) -> None:
    RUN_BASE.mkdir(exist_ok=True)
    APP_STATE_PATH.write_text(
        json.dumps(
            {
                "current_workspace": state.current_workspace,
                "recent_workspaces": state.recent_workspaces,
            },
            indent=2,
        )
    )


def is_valid_workspace(path: str | Path) -> bool:
    p = Path(path).expanduser().resolve()
    return p.is_dir() and (p / "train.py").exists()


def list_recent_workspaces() -> AppState:
    state = _load()
    state.recent_workspaces = [w for w in state.recent_workspaces if Path(w).is_dir()]
    if state.current_workspace and not Path(state.current_workspace).is_dir():
        state.current_workspace = None
    return state


def open_workspace(path: str | Path) -> AppState:
    p = Path(path).expanduser().resolve()
    if not is_valid_workspace(p):
        raise ValueError(f"Not a valid workspace (missing train.py): {p}")
    state = _load()
    s = str(p)
    state.current_workspace = s
    state.recent_workspaces = [s] + [w for w in state.recent_workspaces if w != s]
    state.recent_workspaces = state.recent_workspaces[:MAX_RECENT]
    _save(state)
    return state


def current_workspace() -> Path:
    state = _load()
    if not state.current_workspace:
        raise RuntimeError("No workspace is open. Call open_workspace() first.")
    p = Path(state.current_workspace)
    if not is_valid_workspace(p):
        raise RuntimeError(f"Open workspace is no longer valid: {p}")
    return p
