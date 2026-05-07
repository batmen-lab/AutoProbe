"""Agentic probe pipeline — 4 explicit stages with forward + backward navigation.

Stage 1: user context -> NLP probe designs + confidence -> user selects one
Stage 2: selected probe -> NLP dev plans + confidence -> user selects one
Stage 3: selected dev plan -> agent writes prober.py + integrates train.py
Stage 4: iterative improvement loop driven by prober.py metric

Each stage has explicit input/output artifacts. Backward navigation cleanly
erases the target stage's outputs plus everything later, leaving inputs intact.
"""

from .state import RunState, Stage, list_runs, load_run, new_run
from .workspace import (
    AppState,
    list_recent_workspaces,
    open_workspace,
    is_valid_workspace,
)

__all__ = [
    "RunState",
    "Stage",
    "list_runs",
    "load_run",
    "new_run",
    "AppState",
    "list_recent_workspaces",
    "open_workspace",
    "is_valid_workspace",
]
