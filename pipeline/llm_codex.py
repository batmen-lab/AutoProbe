"""Codex CLI implementations of nlp_call / agent_call / cancel_current.

Only imported when LLM_BACKEND=codex (see the trailer in pipeline/llm.py).
The rest of the pipeline (stages.py, state.py, workspace.py) is unchanged
and unaware which backend is in use.

NLP_MODEL: gpt-5.4 (non-codex, supports clean JSON output under read-only sandbox)
AGENT_MODEL: gpt-5.3-codex (codex variant, native to Codex CLI's shell/apply_patch tools)

These are the two models confirmed to work under a ChatGPT-account subscription.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

NLP_MODEL = "gpt-5.4"
AGENT_MODEL = "gpt-5.3-codex"


# ── Cancellation registry (private to codex backend) ─────────────────────────
_current_proc: subprocess.Popen | None = None
_current_proc_lock = threading.Lock()


def _register(p: subprocess.Popen) -> None:
    global _current_proc
    with _current_proc_lock:
        _current_proc = p


def _unregister() -> None:
    global _current_proc
    with _current_proc_lock:
        _current_proc = None


def cancel_current() -> bool:
    with _current_proc_lock:
        p = _current_proc
    if p is None or p.poll() is not None:
        return False
    try:
        p.kill()
    except ProcessLookupError:
        return False
    return True


# ── Reuse the claude-side truncate + JSON parser (no duplication) ────────────
from .llm import _truncate, _open_log, _parse_json_response  # noqa: E402


# ── Codex JSONL event formatter ──────────────────────────────────────────────
def _format_event(event: dict) -> list[str]:
    out: list[str] = []
    t = event.get("type", "")

    if t == "thread.started":
        out.append(f"[system] thread={event.get('thread_id', '?')}")
        return out
    if t == "turn.started":
        out.append("[turn] started")
        return out
    if t == "turn.completed":
        usage = event.get("usage") or {}
        bits = [f"in={usage.get('input_tokens', 0)}", f"out={usage.get('output_tokens', 0)}"]
        if "cached_input_tokens" in usage:
            bits.append(f"cache={usage['cached_input_tokens']}")
        out.append(f"[done] subtype=success {' '.join(bits)}")
        return out
    if t in ("error", "turn.failed"):
        msg = event.get("message") or (event.get("error") or {}).get("message") or json.dumps(event)
        out.append(f"[error] {_truncate(str(msg), 2000)}")
        return out
    if t == "item.completed":
        item = event.get("item") or {}
        kind = item.get("type", "?")
        if kind == "agent_message":
            txt = (item.get("text") or "").strip()
            if txt:
                out.append(f"[assistant]\n{_truncate(txt)}")
        elif kind == "command_execution":
            cmd = item.get("command") or item.get("input") or {}
            out.append(f"[tool→ shell]\n{_truncate(json.dumps(cmd, ensure_ascii=False), 1200)}")
            res = item.get("aggregated_output") or item.get("output") or ""
            if res:
                out.append(f"[tool←]\n{_truncate(str(res), 2000)}")
        elif kind == "file_change":
            changes = item.get("changes") or item.get("patch") or item
            out.append(f"[tool→ apply_patch]\n{_truncate(json.dumps(changes, ensure_ascii=False), 1200)}")
        elif kind == "reasoning":
            txt = (item.get("text") or item.get("summary") or "").strip()
            if txt:
                out.append(f"[thinking]\n{_truncate(txt)}")
        else:
            out.append(f"[item {kind}] {_truncate(json.dumps(item, ensure_ascii=False), 600)}")
        return out

    out.append(f"[{t}] {_truncate(json.dumps(event, ensure_ascii=False), 400)}")
    return out


_JSON_LINE_PREFIX = re.compile(r"^\s*\{")


def _stream_codex(
    cmd: list[str],
    *,
    cwd: Path | None,
    log_path: Path | None,
    label: str,
) -> tuple[str, str]:
    """Run codex exec --json, tail JSONL events into log_path. Mirror of pipeline/llm._stream_claude."""
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    _register(p)

    f = _open_log(log_path)
    if f:
        f.write(f"\n══ {label} ══\n")

    last_agent_text = ""
    try:
        assert p.stdout is not None
        for raw in p.stdout:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            if not _JSON_LINE_PREFIX.match(raw):
                if f:
                    f.write(f"[trace] {_truncate(raw, 400)}\n")
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                if f:
                    f.write(f"[raw] {_truncate(raw, 1000)}\n")
                continue

            if event.get("type") == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    last_agent_text = item.get("text") or last_agent_text

            if f:
                for line in _format_event(event):
                    f.write(line + "\n")
        p.wait()
    finally:
        _unregister()
        if f:
            f.close()

    stderr = p.stderr.read() if p.stderr else ""
    if p.returncode != 0:
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as ff:
                ff.write(f"[error] exit={p.returncode}\n")
                if stderr:
                    ff.write(_truncate(stderr, 2000) + "\n")
        raise subprocess.CalledProcessError(p.returncode, p.args, last_agent_text, stderr)
    return last_agent_text, stderr


# ── Public surface (matches pipeline.llm signatures exactly) ─────────────────
def nlp_call(
    message: str,
    *,
    model: str = NLP_MODEL,
    log_path: Path | None = None,
    label: str = "nlp_call",
) -> dict:
    """Read-only-sandbox codex call. Final agent message parsed as JSON."""
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="codex_last_", suffix=".txt")
    os.close(tmp_fd)
    try:
        cmd = [
            "codex", "exec",
            "--model", model,
            "--json",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-last-message", tmp_path,
            message,
        ]
        from_stream, _stderr = _stream_codex(
            cmd, cwd=None, log_path=log_path, label=f"{label} → {model}",
        )
        try:
            from_file = Path(tmp_path).read_text()
        except FileNotFoundError:
            from_file = ""
        final = from_file.strip() or from_stream.strip()
        if not final:
            raise RuntimeError(f"{label}: empty response from model")
        return _parse_json_response(final)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def agent_call(
    prompt: str,
    *,
    cwd: Path,
    log_path: Path | None = None,
    model: str = AGENT_MODEL,
    label: str = "agent_call",
) -> None:
    """Workspace-write codex call with approvals bypassed. Native codex tool surface."""
    cmd = [
        "codex", "exec",
        "--model", model,
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--ephemeral",
        "-C", str(cwd),
        prompt,
    ]
    _stream_codex(cmd, cwd=cwd, log_path=log_path, label=f"{label} → {model}  cwd={cwd}")
