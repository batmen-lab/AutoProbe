"""Local-git snapshot manager for train.py.

Replaces the file-copy snapshot scheme (.agent_probe/snapshot/train_version_N.py)
with a bare git repo at .agent_probe/snapshot.git/. Every command runs as
`git --git-dir=<our repo> --work-tree=<workspace>` so the workspace's own
`.git/` (if the user's project is a git repo) is untouched.

Tags we use:
    baseline       → original train.py from new_run (also re-pointed on
                     "keep & re-baseline" after a PASS)
    pre-iter       → train.py going INTO the first stage-4 iteration:
                       - normal mode: after stage-3 PROMPT_FIVE implement
                       - auto-research: after auto_research_setup
    round-N-post   → train.py after round N's edits + training (the state
                     that produced probe_result_N). N starts at 1 (stage-3
                     first run); subsequent rounds increment.

Only train.py is tracked. prober.py and everything else are file-system
concerns handled separately (prober.py is created in stage 3 and never
edited again; revert_to() deletes it directly).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


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


def commit_train(
    workspace: Path,
    message: str,
    tag: str | None = None,
) -> str:
    """Stage train.py and commit it. Optionally (re)point a tag. Returns SHA.

    `--allow-empty` so we still get a commit when train.py is byte-identical
    to the previous one (common for the round-1-post / pre-iter pair).
    """
    train = workspace / "train.py"
    if not train.exists():
        raise FileNotFoundError(f"train.py missing: {train}")
    if not has_repo(workspace):
        # Auto-initialize instead of crashing: reverting to an early stage wipes
        # the workspace's .agent_probe/ (snapshot.git included), and the next
        # implement would otherwise fail with "snapshot.git not initialized".
        init(workspace)
    _git(workspace, "add", "--", "train.py")
    _git(workspace, "commit", "--allow-empty", "--quiet", "-m", message)
    sha = _git(workspace, "rev-parse", "HEAD").stdout.strip()
    if tag:
        # -f so later rounds can re-point a tag (e.g. baseline on re-baseline).
        _git(workspace, "tag", "-f", tag, sha)
    return sha


def restore_train(workspace: Path, ref: str) -> None:
    """Restore train.py from a commit / tag / branch. No-op if ref missing."""
    if not has_repo(workspace):
        return
    if not rev_parse(workspace, ref):
        return
    _git(workspace, "checkout", ref, "--", "train.py")


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
