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
import re
import subprocess
import threading
import time
from pathlib import Path

NLP_MODEL = "opus"
AGENT_MODEL = "opus"
AGENT_MAX_RETRIES = 2          # extra attempts after the first (3 total)
AGENT_RETRY_BACKOFF_S = 4.0    # pause before re-invoking the agent on a transient failure


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
# Read-only + search tools exposed to the JSON-generating NLP calls. The probe
# and dev-plan prompts invite the model to "survey peer work" and inspect the
# codebase; with an empty tool list those tool calls bounced as "No such tool
# available" and could derail the whole response (e.g. a stray </invoke> tag).
# We allow search/read tools but NOT Write/Edit/Bash so these calls stay
# side-effect-free. Tool EXECUTION is performed by the claude CLI itself, not
# by the model — so this works regardless of which model ccr routes to.
#
# NOTE: WebSearch is intentionally EXCLUDED. It is Anthropic's *server-side*
# search; routed through ccr → OpenRouter → deepseek it executes but returns
# empty results, so the model loops on it fruitlessly. WebFetch (a plain
# client-side HTTP GET) + the local search tools (Grep/Glob/Read) DO work and
# cover "find a similar project in the repo" + "read a known reference URL".
# To get real open-web search, route these calls to a web-capable model
# (e.g. OpenRouter's `<model>:online` variant) or to a Claude model directly.
NLP_TOOLS = "WebFetch,Read,Grep,Glob"
NLP_MAX_RETRIES = 2  # extra attempts after the first (3 total)


def nlp_call(
    message: str,
    *,
    model: str = NLP_MODEL,
    log_path: Path | None = None,
    label: str = "nlp_call",
    tools: str = NLP_TOOLS,
) -> dict:
    """Call the NLP model and parse its JSON response.

    Exposes read-only + search tools so the model can ground its answer, and
    retries on transient failures — malformed/non-JSON output (incl. the stray
    tool-call-tag derailment), empty responses, and provider errors — each time
    with a reinforced "JSON only, no tool calls" nudge. A subprocess killed by a
    signal (i.e. user cancellation) is NOT retried. Streams events to log_path.
    """
    base = [
        "claude",
        "-p",
        "--model",
        model,
        "--tools",
        tools,
        "--dangerously-skip-permissions",  # headless: don't block on tool prompts
        "--no-session-persistence",
    ]
    last_err: Exception | None = None
    for attempt in range(NLP_MAX_RETRIES + 1):
        msg = message
        if attempt > 0:
            msg = (
                message
                + "\n\nIMPORTANT: Respond with ONLY the JSON object — no prose, no "
                "markdown fences, and no tool-call tags. Do not emit any tool call "
                "in your final message."
            )
        lbl = f"{label} → {model}" + (
            f"  (retry {attempt}/{NLP_MAX_RETRIES})" if attempt else ""
        )
        try:
            final_text, _stderr = _stream_claude(base + [msg], log_path=log_path, label=lbl)
            if not final_text.strip():
                raise RuntimeError(f"{label}: empty response from model")
            return _parse_json_response(final_text)
        except subprocess.CalledProcessError as e:
            # Negative returncode = killed by signal (user cancel) — don't retry.
            if e.returncode is not None and e.returncode < 0:
                raise
            last_err = e
        except (json.JSONDecodeError, RuntimeError) as e:
            last_err = e
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as ff:
                ff.write(
                    f"[retry] {label}: attempt {attempt + 1} failed: "
                    f"{type(last_err).__name__}: {_truncate(str(last_err), 300)}\n"
                )
    assert last_err is not None
    raise last_err


# Some routed models (qwen via OpenRouter, etc.) ignore "return only the JSON"
# instructions and emit preamble + ```json ... ``` fences. Be tolerant.
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
# Typographic quotes weaker models emit instead of ASCII ones.
_SMART_QUOTES = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'",
}


def _repair_json(s: str) -> str:
    """Best-effort repair of almost-JSON emitted by weaker/routed models.

    Handles the failure modes we actually see from OpenRouter-routed models:
      • unescaped double quotes *inside* string values
        (e.g. ``the "gold standard" approach`` — the crash that wedged a run)
      • literal newlines/tabs inside string values
      • trailing commas before ``}``/``]``
      • typographic “smart” quotes

    A single state-machine pass walks the text tracking string context. A ``"``
    inside a string is treated as the closing delimiter only when the next
    non-whitespace char is structural (``,`` ``}`` ``]`` ``:`` or EOF);
    otherwise it is an inner quote and gets escaped. This is a heuristic — it
    can misjudge prose like ``"he said "hi", bye"`` — so it runs ONLY as a
    last-resort fallback, after every strict parse attempt has failed.
    """
    out: list[str] = []
    in_str = False
    esc = False
    n = len(s)
    i = 0
    while i < n:
        c = s[i]
        if not in_str:
            if c == '"':
                in_str = True
            out.append(c)
            i += 1
            continue
        # inside a string
        if esc:
            out.append(c)
            esc = False
            i += 1
            continue
        if c == "\\":
            out.append(c)
            esc = True
            i += 1
            continue
        if c in _SMART_QUOTES:
            out.append("\\" + '"' if _SMART_QUOTES[c] == '"' else _SMART_QUOTES[c])
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            nxt = s[j] if j < n else ""
            if nxt in ",}]:" or nxt == "":
                out.append('"')
                in_str = False
            else:
                out.append('\\"')  # inner quote → escape
            i += 1
            continue
        if c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        else:
            out.append(c)
        i += 1
    repaired = "".join(out)
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    return repaired


def _candidates(s: str):
    """Yield JSON-ish substrings to try, widest-confidence first."""
    yield s
    m = _FENCE_RE.search(s)
    if m:
        yield m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        end = s.rfind(closer)
        if start != -1 and end > start:
            yield s[start : end + 1]


def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from a model response, tolerating common breakage.

    Strategy: try strict ``json.loads`` on each candidate substring (whole
    text, fenced block, outermost braces). Only if every strict attempt fails
    do we run ``_repair_json`` over the candidates and retry. Well-formed
    responses never touch the repair path.
    """
    s = (text or "").strip()
    cands = list(_candidates(s))
    # Pass 1 — strict.
    for cand in cands:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # Pass 2 — repair fallback.
    for cand in cands:
        try:
            return json.loads(_repair_json(cand))
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError(
        f"NLP response not valid JSON even after repair (first 200 chars): {s[:200]!r}",
        s[:1],
        0,
    )


def agent_call(
    prompt: str,
    *,
    cwd: Path,
    log_path: Path | None = None,
    model: str = AGENT_MODEL,
    label: str = "agent_call",
) -> None:
    """Run the code agent inside `cwd`. Streams events to log_path.

    Retries on transient subprocess failures (provider 5xx / idle-timeout / API
    errors make the claude CLI exit non-zero with a positive code), with a short
    backoff. A subprocess killed by a signal (user cancellation -> negative
    returncode) is NOT retried. The agent re-reads the workspace on each attempt,
    so a retry safely re-does the implementation rather than resuming a partial
    one. Raises subprocess.CalledProcessError if all attempts fail.
    """
    full_cmd = [
        "claude", "-p", "--dangerously-skip-permissions",
        "--model", model, prompt,
        "--output-format", "stream-json", "--verbose",
    ]

    def _attempt(lbl: str) -> None:
        # cwd-aware Popen: the working dir matters for the agent.
        p = subprocess.Popen(
            full_cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        _register(p)
        f = _open_log(log_path)
        if f:
            f.write(f"\n══ {lbl} → {model}  cwd={cwd} ══\n")
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

    for attempt in range(AGENT_MAX_RETRIES + 1):
        lbl = label + (f"  (retry {attempt}/{AGENT_MAX_RETRIES})" if attempt else "")
        try:
            _attempt(lbl)
            return
        except subprocess.CalledProcessError as e:
            # Negative returncode = killed by signal (user cancel) -> don't retry.
            if e.returncode is not None and e.returncode < 0:
                raise
            if attempt >= AGENT_MAX_RETRIES:
                raise
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as ff:
                    ff.write(
                        f"[retry] {label}: attempt {attempt + 1} failed "
                        f"(exit={e.returncode}) — retrying in {AGENT_RETRY_BACKOFF_S:g}s...\n"
                    )
            time.sleep(AGENT_RETRY_BACKOFF_S)


# ── Backend swap (codex CLI) ────────────────────────────────────────────────
# When LLM_BACKEND=codex is set in the environment, swap the three public
# entry points to the codex CLI implementations. Default is "claude" — the
# code above runs unchanged. stages.py / state.py / server/app.py never need
# to know which backend is active.
import os as _os  # noqa: E402

if _os.environ.get("LLM_BACKEND", "claude").lower() == "codex":
    from . import llm_codex as _codex
    nlp_call = _codex.nlp_call
    agent_call = _codex.agent_call
    cancel_current = _codex.cancel_current
    NLP_MODEL = _codex.NLP_MODEL
    AGENT_MODEL = _codex.AGENT_MODEL
