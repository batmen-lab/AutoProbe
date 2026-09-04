#!/usr/bin/env python3
"""
CCR guard - runs every 60s from a systemd timer.

Two jobs:

  1. UNDO GLOBAL PROFILE TAKEOVER. CCR can rewrite ~/.claude/settings.json (and
     ~/.codex/config.toml) to point at 127.0.0.1:3456. When CCR then stops,
     Claude Code has no endpoint: any agent running on this box goes mute
     mid-task and cannot repair itself, because the tool it would repair itself
     with is the thing that is broken. This has happened twice. The guard
     detects it and restores the known-good global config.

  2. KEEP THE GATEWAY UP. CCR has autoStart=false and does not resurrect itself.
     A dead gateway does not fail loudly - pipeline/router.py degrades to
     pass-through and the run silently bills Anthropic at full price, which is
     the exact thing this setup exists to avoid.

The guard NEVER touches <repo>/.claude/settings.local.json. That file is the
legitimate, per-repo routing config and is the whole point of the setup.

To pause it (e.g. for manual maintenance):
    touch ~/.claude-code-router/guard.disabled
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request

HOME = os.path.expanduser("~")
CCR_DIR = os.path.join(HOME, ".claude-code-router")
DB = os.path.join(CCR_DIR, "config.sqlite")
GOLDEN = os.path.join(CCR_DIR, "golden-claude-settings.json")
PAUSE = os.path.join(CCR_DIR, "guard.disabled")
LOG = os.path.join(CCR_DIR, "guard.log")
CLAUDE_SETTINGS = os.path.join(HOME, ".claude", "settings.json")
GATEWAY = "http://127.0.0.1:3456"
MAX_LOG_BYTES = 1_000_000

ROUTER_ENV_KEYS = [
    "ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE_URL", "CLAUDE_AGENT_API_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
]

_actions = []


def log(msg, action=False):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        if os.path.exists(LOG) and os.path.getsize(LOG) > MAX_LOG_BYTES:
            os.replace(LOG, LOG + ".1")
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if action:
        _actions.append(msg)
        # Make repairs visible in journalctl/syslog, not just our own log file.
        subprocess.run(["logger", "-t", "ccr-guard", msg], check=False)


def sh(cmd, check=False):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          check=check)


def health_ok(timeout=4):
    try:
        with urllib.request.urlopen(GATEWAY + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ccr_stop():
    sh("ccr stop")
    sh("pkill -f 'claude-code-router/dist/main/cli.js'")
    sh("pkill -f gateway-bootstrap.js")
    time.sleep(1)


def ccr_start(wait=25):
    sh("ccr start --no-open")
    for _ in range(wait):
        if health_ok():
            return True
        time.sleep(1)
    return False


# ------------------------------------------------------------ takeover checks

def settings_hijacked():
    if not os.path.exists(CLAUDE_SETTINGS):
        return []
    try:
        with open(CLAUDE_SETTINGS) as f:
            data = json.load(f)
    except Exception:
        return ["<unparseable>"]
    env = data.get("env") or {}
    return [k for k in ROUTER_ENV_KEYS if k in env]


def fix_settings():
    """Restore the golden global config. Golden is written by provision_ccr.py."""
    if os.path.exists(GOLDEN):
        shutil.copyfile(GOLDEN, CLAUDE_SETTINGS)
        log("RESTORED ~/.claude/settings.json from golden copy", action=True)
        return
    # No golden copy: fall back to CCR's own pre-takeover backup, else strip.
    orig = CLAUDE_SETTINGS + ".ccr-original"
    if os.path.exists(orig):
        shutil.copyfile(orig, CLAUDE_SETTINGS)
        log("RESTORED ~/.claude/settings.json from .ccr-original", action=True)
        return
    try:
        with open(CLAUDE_SETTINGS) as f:
            data = json.load(f)
    except Exception:
        data = {}
    env = data.get("env") or {}
    for k in ROUTER_ENV_KEYS:
        env.pop(k, None)
    if env:
        data["env"] = env
    else:
        data.pop("env", None)
    with open(CLAUDE_SETTINGS, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    log("STRIPPED router env from ~/.claude/settings.json (no golden copy)",
        action=True)


def profile_violations():
    """Return a description of any enabled/global profile settings in the DB."""
    if not os.path.exists(DB):
        return []
    bad = []
    try:
        # Read a copy so a live WAL cannot trip us up.
        tmp = "/tmp/ccr-guard-read-%d.sqlite" % os.getpid()
        for ext in ("", "-wal", "-shm"):
            if os.path.exists(DB + ext):
                shutil.copyfile(DB + ext, tmp + ext)
        con = sqlite3.connect(tmp)
        row = con.execute(
            "select value_json from app_config where key='default'").fetchone()
        con.close()
        for ext in ("", "-wal", "-shm"):
            if os.path.exists(tmp + ext):
                os.remove(tmp + ext)
        if not row:
            return []
        prof = json.loads(row[0]).get("profile", {})
        if prof.get("enabled"):
            bad.append("profile.enabled")
        for sub in ("claudeCode", "codex"):
            if isinstance(prof.get(sub), dict) and prof[sub].get("enabled"):
                bad.append("profile.%s.enabled" % sub)
        for p in prof.get("profiles", []):
            if p.get("enabled"):
                bad.append("profiles[%s].enabled" % p.get("id"))
            if p.get("scope") == "global":
                bad.append("profiles[%s].scope=global" % p.get("id"))
    except Exception as e:
        log("could not read config DB: %s" % e)
    return bad


def fix_profiles():
    """Force every profile switch off. Requires ccr stopped to write safely."""
    con = sqlite3.connect(DB)
    try:
        row = con.execute(
            "select value_json from app_config where key='default'").fetchone()
        if not row:
            return
        cfg = json.loads(row[0])
        prof = cfg.setdefault("profile", {})
        prof["enabled"] = False
        for sub in ("claudeCode", "codex"):
            if isinstance(prof.get(sub), dict):
                prof[sub]["enabled"] = False
        for p in prof.get("profiles", []):
            p["enabled"] = False
            p["scope"] = "manual"
        con.execute("update app_config set value_json=?, updated_at=? "
                    "where key='default'",
                    (json.dumps(cfg), time.strftime("%Y-%m-%dT%H:%M:%SZ")))
        con.commit()
        con.execute("pragma wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    log("DISABLED profile takeover in config.sqlite", action=True)


def purge_artifacts():
    tm = os.path.join(CCR_DIR, "global-profile-takeover.json")
    if os.path.exists(tm):
        os.remove(tm)
        log("REMOVED global-profile-takeover.json", action=True)
    for f in (os.path.join(HOME, ".codex", "config.toml"),
              os.path.join(HOME, ".codex", "ccr-model-catalog.json"),
              os.path.join(HOME, ".codex", "claude-code-router.config.toml"),
              os.path.join(HOME, ".claude", ".claude.json")):
        if os.path.exists(f + ".ccr-original-missing") and os.path.exists(f):
            os.remove(f)
            os.remove(f + ".ccr-original-missing")
            log("REMOVED CCR-created %s" % f, action=True)


# ---------------------------------------------------------------------- main

def main():
    if os.path.exists(PAUSE):
        log("paused (%s exists) - doing nothing" % PAUSE)
        return 0

    hijacked = settings_hijacked()
    violations = profile_violations()
    needs_restart = False

    if hijacked or violations:
        log("TAKEOVER DETECTED settings=%s db=%s"
            % (hijacked or "clean", violations or "clean"), action=True)
        ccr_stop()
        needs_restart = True
        if hijacked:
            fix_settings()
        if violations:
            fix_profiles()
        purge_artifacts()
    else:
        purge_artifacts()

    if needs_restart or not health_ok():
        if not needs_restart:
            log("gateway not healthy - starting it", action=True)
        if ccr_start():
            log("gateway healthy at %s" % GATEWAY)
        else:
            log("FAILED to bring the gateway up", action=True)
            return 1

    if not _actions:
        log("ok (settings clean, profiles off, gateway healthy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
