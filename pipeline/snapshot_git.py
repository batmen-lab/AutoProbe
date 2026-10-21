"""Local-git snapshot manager for the workspace.

Replaces the file-copy snapshot scheme (.agent_probe/snapshot/train_version_N.py)
with a bare git repo at .agent_probe/snapshot.git/. Every command runs as
`git --git-dir=<our repo> --work-tree=<workspace>` so the workspace's own
`.git/` (if the user's project is a git repo) is untouched.

Tags we use:
    baseline       → original workspace from new_run (also re-pointed on
                     "keep & re-baseline" after a PASS)
    pre-iter       → workspace going INTO the first stage-4 iteration:
                       - normal mode: after stage-3 PROMPT_FIVE implement
                       - auto-research: after auto_research_setup
    round-N-post   → workspace after round N's keep/revert decision (the state
                     the next round starts from). N starts at 1 (stage-3
                     first run); subsequent rounds increment.
    round-N-tried  → workspace exactly as round N's edits + training left it,
                     BEFORE the keep/revert decision. Lets the user keep a
                     reverted round afterwards.

Scope: the WHOLE workspace except `.agent_probe/` (and anything the
workspace's own .gitignore ignores). Restoring a ref makes the tracked tree
match it exactly: files the ref has are written back, tracked files it lacks
are deleted.

Repos created before the whole-workspace scope (no `autoprobe.scope` key in
their config) keep the old train.py-only behaviour, so restoring one of their
train.py-only commits can never delete the rest of an existing workspace.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCOPE_KEY = "autoprobe.scope"
SCOPE_WORKSPACE = "workspace"

# Never snapshotted: AutoProbe's own state (snapshot.git lives inside it) and
# interpreter caches. The workspace's own .gitignore is honoured on top.
_EXCLUDES = (".agent_probe/", "__pycache__/", "*.pyc")


def _git_dir(workspace: Path) -> Path:
    return workspace / ".agent_probe" / "snapshot.git"


def _git(
    workspace: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run git against our private repo + workspace as work-tree."""
    cmd = [
        "git",
        "--git-dir", str(_git_dir(workspace)),
        "--work-tree", str(workspace),
        *args,
    ]
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def has_repo(workspace: Path) -> bool:
    return (_git_dir(workspace) / "HEAD").exists()


def _whole_workspace(workspace: Path) -> bool:
    """True for repos created with the whole-workspace scope."""
    r = subprocess.run(
        ["git", "--git-dir", str(_git_dir(workspace)), "config", "--get", SCOPE_KEY],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == SCOPE_WORKSPACE


def init(workspace: Path) -> None:
    """Idempotent: create the bare repo if it doesn't already exist."""
    gd = _git_dir(workspace)
    if (gd / "HEAD").exists():
        return
    gd.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(gd)],
        check=True, capture_output=True,
    )
    # Per-repo identity so commits succeed without depending on global config.
    subprocess.run(
        ["git", "--git-dir", str(gd), "config", "user.email", "agent-probe@local"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "--git-dir", str(gd), "config", "user.name", "Agent Probe"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "--git-dir", str(gd), "config", SCOPE_KEY, SCOPE_WORKSPACE],
        check=True, capture_output=True,
    )
    info = gd / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "exclude").write_text("\n".join(_EXCLUDES) + "\n")


def _stage(workspace: Path) -> None:
    """Stage the snapshot scope into the index."""
    if _whole_workspace(workspace):
        _git(workspace, "add", "-A", "--", ".")
    else:
        _git(workspace, "add", "--", "train.py")


def commit_train(
    workspace: Path,
    message: str,
    tag: str | None = None,
) -> str:
    """Snapshot the workspace (train.py only for legacy repos) and commit it.
    Optionally (re)point a tag. Returns SHA.

    `--allow-empty` so we still get a commit when nothing changed since the
    previous one (common for the round-1-post / pre-iter pair).
    """
    train = workspace / "train.py"
    if not train.exists():
        raise FileNotFoundError(f"train.py missing: {train}")
    if not has_repo(workspace):
        # Auto-initialize instead of crashing: reverting to an early stage wipes
        # the workspace's .agent_probe/ (snapshot.git included), and the next
        # implement would otherwise fail with "snapshot.git not initialized".
        init(workspace)
    _stage(workspace)
    _git(workspace, "commit", "--allow-empty", "--quiet", "-m", message)
    sha = _git(workspace, "rev-parse", "HEAD").stdout.strip()
    if tag:
        # -f so later rounds can re-point a tag (e.g. baseline on re-baseline).
        _git(workspace, "tag", "-f", tag, sha)
    return sha


def restore_train(workspace: Path, ref: str) -> None:
    """Restore the workspace (train.py only for legacy repos) from a commit /
    tag / branch. No-op if ref missing."""
    if not has_repo(workspace):
        return
    if not rev_parse(workspace, ref):
        return
    if not _whole_workspace(workspace):
        _git(workspace, "checkout", ref, "--", "train.py")
        return
    # Stage the current tree first so files created since `ref` are in the
    # index; `read-tree -u --reset` then deletes every indexed file `ref`
    # lacks and writes back every file it has.
    _stage(workspace)
    _git(workspace, "read-tree", "-u", "--reset", ref)


def tag_exists(workspace: Path, tag: str) -> bool:
    if not has_repo(workspace):
        return False
    r = _git(
        workspace,
        "rev-parse", "--verify", "--quiet",
        f"refs/tags/{tag}",
        check=False,
    )
    return r.returncode == 0


def rev_parse(workspace: Path, ref: str) -> str | None:
    """Return the SHA `ref` resolves to, or None if it doesn't exist."""
    if not has_repo(workspace):
        return None
    r = _git(workspace, "rev-parse", "--verify", "--quiet", ref, check=False)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def log_shas(workspace: Path) -> list[str]:
    """All commit SHAs, newest first. Empty if no repo / no commits."""
    if not has_repo(workspace):
        return []
    r = _git(workspace, "log", "--pretty=format:%H", check=False)
    if r.returncode != 0 or not r.stdout:
        return []
    return r.stdout.split("\n")


def delete_repo(workspace: Path) -> None:
    """Wipe the entire snapshot history (called from full reverts that also
    reset the iteration ledger). The caller is responsible for re-initing
    afterwards if needed.
    """
    import shutil
    gd = _git_dir(workspace)
    if gd.exists():
        shutil.rmtree(gd, ignore_errors=True)
