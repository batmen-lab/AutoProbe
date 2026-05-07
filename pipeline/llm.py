"""Subprocess wrappers for the `claude` CLI used as NLP and as a code agent.

Two flavors:
- nlp_call: short, JSON-returning, no tools, no session persistence
- agent_call: long-running, allowed to edit files in the workspace

Both run claude in `--output-format stream-json --verbose` so we can stream
thinking blocks, tool uses, tool results, and assistant text into the per-run
log file as they happen. The frontend's SSE log dock surfaces this live.

Both also register their Popen handle so the API server can cancel an in-flight
stage action via cancel_current().
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

NLP_MODEL = "opus"
AGENT_MODEL = "opus"


# ── Cancellation registry ────────────────────────────────────────────────────
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
    """Kill the active subprocess if any. Returns True iff a process was killed."""
    with _current_proc_lock:
        p = _current_proc
    if p is None or p.poll() is not None:
        return False
    try:
        p.kill()
    except ProcessLookupError:
        return False
    return True


# ── Stream parser ────────────────────────────────────────────────────────────
_BLOCK_CAP = 8000  # chars per emitted block; long thinking blocks get truncated


def _truncate(s: str, cap: int = _BLOCK_CAP) -> str:
    if len(s) <= cap:
        return s
    return s[:cap] + f"\n…[+{len(s) - cap} chars truncated]"


_NOISY_EVENT_TYPES = {"rate_limit_event"}


def _format_event(event: dict) -> list[str]:
    """Translate one stream-json event into one or more human-readable log lines."""
    out: list[str] = []
    t = event.get("type")

    if t in _NOISY_EVENT_TYPES:
        return out

    if t == "system":
        sub = event.get("subtype", "")
        if sub == "init":
            model = event.get("model") or "?"
            tools = event.get("tools") or []
            out.append(f"[system] init  model={model}  tools={len(tools)}")
        return out

    if t == "assistant":
        msg = event.get("message", {})
        for block in msg.get("content", []) or []:
            bt = block.get("type")
            if bt == "thinking":
                txt = (block.get("thinking") or "").strip()
                if txt:
                    out.append(f"[thinking]\n{_truncate(txt)}")
            elif bt == "text":
                txt = (block.get("text") or "").strip()
                if txt:
                    out.append(f"[assistant]\n{_truncate(txt)}")
            elif bt == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                preview = json.dumps(inp, ensure_ascii=False)
                out.append(f"[tool→ {name}]\n{_truncate(preview, 1200)}")
        return out

    if t == "user":
        msg = event.get("message", {})
        for block in msg.get("content", []) or []:
            if block.get("type") == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                content = str(content).strip()
                if content:
                    out.append(f"[tool←]\n{_truncate(content, 2000)}")
        return out

    if t == "result":
        sub = event.get("subtype", "")
        usage = event.get("usage") or {}
        cost = event.get("total_cost_usd")
        bits = [f"subtype={sub}"]
        if usage:
            bits.append(f"in={usage.get('input_tokens', 0)}")
            bits.append(f"out={usage.get('output_tokens', 0)}")
        if cost is not None:
            bits.append(f"cost=${cost:.4f}")
        out.append(f"[done] {' '.join(bits)}")
        return out

    # Unknown event types — write a compact one-liner so we don't lose visibility.
    out.append(f"[{t}] {_truncate(json.dumps(event, ensure_ascii=False), 400)}")
    return out


def _open_log(log_path: Path | None):
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("a", encoding="utf-8", buffering=1)  # line-buffered


def _stream_claude(
    cmd: list[str],
    *,
    log_path: Path | None,
    label: str,
) -> tuple[str, str]:
    """Run `claude` with stream-json output and tail the events to log_path.

    Returns (final_text, stderr). The final_text is the canonical assistant
    output (from the result event), suitable for JSON parsing in nlp_call.
    Raises subprocess.CalledProcessError on non-zero exit.
    """
    full_cmd = list(cmd) + ["--output-format", "stream-json", "--verbose"]
    p = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered so we get events as they arrive
    )
    _register(p)

    f = _open_log(log_path)
    if f:
        f.write(f"\n══ {label} ══\n")

    final_text = ""
    try:
        assert p.stdout is not None
        for raw in p.stdout:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                if f:
                    f.write(f"[raw] {_truncate(raw, 1000)}\n")
                continue

            # Capture the canonical final result for nlp_call to parse.
            if event.get("type") == "result" and event.get("subtype") == "success":
                final_text = event.get("result", "") or ""

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
        raise subprocess.CalledProcessError(p.returncode, p.args, final_text, stderr)
    return final_text, stderr


# ── Calls ────────────────────────────────────────────────────────────────────
def nlp_call(
    message: str,
    *,
    model: str = NLP_MODEL,
    log_path: Path | None = None,
    label: str = "nlp_call",
) -> dict:
    """Call the NLP model and parse its JSON response.

    Streams thinking + assistant text into log_path while the call is running.
    """
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--tools",
        "",
        "--no-session-persistence",
        message,
    ]
    final_text, _stderr = _stream_claude(cmd, log_path=log_path, label=f"{label} → {model}")
    if not final_text.strip():
        raise RuntimeError(f"{label}: empty response from model")
    return json.loads(final_text)


def agent_call(
    prompt: str,
    *,
    cwd: Path,
    log_path: Path | None = None,
    model: str = AGENT_MODEL,
    label: str = "agent_call",
) -> None:
    """Run the code agent inside `cwd`. Streams events to log_path."""
    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--model",
        model,
        prompt,
    ]
    # cwd-aware Popen: we hand-roll the equivalent of _stream_claude because
    # the working dir matters for the agent.
    full_cmd = list(cmd) + ["--output-format", "stream-json", "--verbose"]
    p = subprocess.Popen(
        full_cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    _register(p)
    f = _open_log(log_path)
    if f:
        f.write(f"\n══ {label} → {model}  cwd={cwd} ══\n")

    try:
        assert p.stdout is not None
        for raw in p.stdout:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                if f:
                    f.write(f"[raw] {_truncate(raw, 1000)}\n")
                continue
            if f:
                for line in _format_event(event):
                    f.write(line + "\n")
        p.wait()
    finally:
        _unregister()
        if f:
            f.close()

    if p.returncode != 0:
        stderr = p.stderr.read() if p.stderr else ""
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as ff:
                ff.write(f"[error] exit={p.returncode}\n")
                if stderr:
                    ff.write(_truncate(stderr, 2000) + "\n")
        raise subprocess.CalledProcessError(p.returncode, p.args, None, stderr)
