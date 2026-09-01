"""claude-code-router (ccr) v3 wiring for the `claude` subprocesses.

Reads ccr's config and hands the resulting environment to every ``claude``
subprocess ``pipeline/llm.py`` spawns, so the pipeline's calls go through the
local gateway instead of straight to Anthropic.

What the code below has to accommodate:

* config lives in SQLite — ``~/.claude-code-router/config.sqlite`` (WAL mode;
  ``app_config`` holds one JSON blob under key ``default``, ``api_keys`` holds
  the gateway bearer key);
* the gateway serves the Anthropic Messages API on ``gateway.port`` (default
  3456) and wants that key in ``x-api-key`` / ``Authorization``;
* the gateway does **not** alias ``claude-*`` model names onto the configured
  route — a request must name a model the provider actually lists, or it 400s
  with "Model ... is not configured for target provider". Hence the
  ``ANTHROPIC_DEFAULT_*_MODEL`` vars: ``llm.py`` asks for ``--model opus`` and
  the CLI resolves the alias before the request leaves the box.

(ccr v1's ``config.json`` and ``ccr activate`` are gone; this module is what
replaced the old ``eval "$(ccr activate)"`` in the Makefile. Don't reintroduce
it — shell-eval only works when the server is launched from that shell.)

Escape hatches
--------------
``AUTOPROBE_ROUTER=off`` disables injection entirely (talk to Anthropic
directly with your own ``ANTHROPIC_API_KEY``). Any ``ANTHROPIC_*`` variable
you export yourself wins over what we read from the DB.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

CCR_HOME = Path(os.environ.get("CCR_HOME", Path.home() / ".claude-code-router"))
CONFIG_DB = CCR_HOME / "config.sqlite"

# Variables we derive from the ccr config. A value already present in the
# real environment is never overwritten — exporting one by hand is the
# documented way to override a single field.
_MANAGED = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
)


class RouterUnavailable(RuntimeError):
    """ccr is installed-but-unreadable, or not installed at all."""


def _read_db() -> tuple[dict, str]:
    """Return (app_config JSON, gateway api key).

    The DB is in WAL mode and ccr holds it open, so we snapshot the trio of
    files into a temp dir and read that. Cheap (a few hundred KB) and it can
    never disturb the live router.
    """
    if not CONFIG_DB.exists():
        raise RouterUnavailable(f"ccr config DB not found at {CONFIG_DB}")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "config.sqlite"
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(CONFIG_DB) + suffix)
            if src.exists():
                shutil.copy(src, str(dst) + suffix)
        con = sqlite3.connect(dst)
        try:
            row = con.execute(
                "SELECT value_json FROM app_config WHERE key = 'default'"
            ).fetchone()
            if not row:
                raise RouterUnavailable("ccr app_config has no 'default' row")
            cfg = json.loads(row[0])
            key_row = con.execute(
                "SELECT encrypted_key FROM api_keys ORDER BY "
                "(id = 'local-gateway') DESC, created_at ASC LIMIT 1"
            ).fetchone()
        finally:
            con.close()
    if not key_row or not key_row[0]:
        raise RouterUnavailable(
            "ccr has no gateway API key — open the ccr UI (`ccr ui`) and create one"
        )
    return cfg, key_row[0]


def _model_of(route: str | None) -> str:
    """``"openrouter,deepseek/deepseek-v4-flash-0731"`` → the model id.

    ccr stores routes as ``<provider>,<model>``; the gateway wants the bare
    model id, which must be one the provider lists.
    """
    if not route:
        return ""
    return route.split(",", 1)[1].strip() if "," in route else route.strip()


def gateway_env() -> dict[str, str]:
    """Env vars that point the `claude` CLI at the local ccr gateway."""
    cfg, api_key = _read_db()

    gw = cfg.get("gateway") or {}
    host = gw.get("host") or cfg.get("HOST") or "127.0.0.1"
    port = gw.get("port") or cfg.get("PORT") or 3456

    router = cfg.get("Router") or {}
    default_model = _model_of(router.get("default"))
    if not default_model:
        raise RouterUnavailable(
            "ccr Router has no default route — set one in the ccr UI (`ccr ui`)"
        )
    # Cheap/fast route, used by the CLI for haiku-class background work
    # (titles, summaries). Falls back to the default route.
    small_model = _model_of(router.get("background")) or default_model
    think_model = _model_of(router.get("think")) or default_model

    return {
        "ANTHROPIC_BASE_URL": f"http://{host}:{port}",
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_MODEL": default_model,
        # llm.py asks for `--model opus`; the CLI resolves the alias through
        # these before the request leaves the box, so the gateway only ever
        # sees a model the provider actually lists.
        "ANTHROPIC_DEFAULT_OPUS_MODEL": think_model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": default_model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": small_model,
        "ANTHROPIC_SMALL_FAST_MODEL": small_model,
    }


_cached: dict[str, str] | None = None
_cache_valid = False


def subprocess_env(refresh: bool = False) -> dict[str, str]:
    """Full environment for a `claude` subprocess.

    Returns ``os.environ`` plus the ccr gateway wiring. Anything you already
    exported wins. With ``AUTOPROBE_ROUTER=off``, or when ccr's config can't
    be read, the environment is passed through untouched so the CLI falls
    back to whatever auth it normally uses.
    """
    global _cached, _cache_valid
    env = dict(os.environ)

    if env.get("AUTOPROBE_ROUTER", "ccr").lower() in ("off", "none", "0", "direct"):
        return env

    if refresh or not _cache_valid:
        try:
            _cached = gateway_env()
        except (RouterUnavailable, sqlite3.Error, json.JSONDecodeError, OSError):
            _cached = None
        _cache_valid = True

    if not _cached:
        return env

    for k, v in _cached.items():
        env.setdefault(k, v)
    # ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are mutually exclusive for the
    # CLI; a stale key in the ambient env silently beats the gateway token.
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)
    return env


def describe() -> str:
    """One-line summary for logs / `make doctor`."""
    if os.environ.get("AUTOPROBE_ROUTER", "ccr").lower() in ("off", "none", "0", "direct"):
        return "router: disabled (AUTOPROBE_ROUTER=off) — talking to Anthropic directly"
    try:
        e = gateway_env()
    except RouterUnavailable as exc:
        return f"router: UNAVAILABLE — {exc}"
    return (
        f"router: ccr {e['ANTHROPIC_BASE_URL']} "
        f"default={e['ANTHROPIC_MODEL']} small={e['ANTHROPIC_SMALL_FAST_MODEL']}"
    )


if __name__ == "__main__":
    import sys

    # `--base-url` / `--port` exist for the Makefile: the gateway port lives in
    # the ccr DB, so shell recipes ask us for it instead of hardcoding 3456.
    if "--base-url" in sys.argv:
        print(gateway_env()["ANTHROPIC_BASE_URL"])
    elif "--port" in sys.argv:
        print(gateway_env()["ANTHROPIC_BASE_URL"].rsplit(":", 1)[-1])
    else:
        print(describe())
