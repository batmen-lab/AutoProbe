#!/usr/bin/env python3
"""
Idempotent CCR provisioning for an AutoProbe worker VM.

Safe to re-run at any time. Every run re-asserts the same invariants, so this
is both the *setup* path and the *repair* path. If an agent has broken the
routing, run this again rather than hand-editing anything.

    export OPENROUTER_API_KEY=sk-or-v1-...
    python3 provision_ccr.py

Options:
    --repo PATH             AutoProbe checkout (default /mnt/workspace/AutoProbe)
    --model ID              OpenRouter model id
    --context-tokens N      real context window of that model
    --install-packages      npm i -g ccr + claude first
    --no-start              configure only, do not start the gateway
"""
import argparse
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import time
import urllib.request

HOME = os.path.expanduser("~")
CCR_DIR = os.path.join(HOME, ".claude-code-router")
DB = os.path.join(CCR_DIR, "config.sqlite")
GOLDEN = os.path.join(CCR_DIR, "golden-claude-settings.json")
# Held while provisioning so the guard timer cannot restart ccr mid-write.
GUARD_PAUSE = os.path.join(CCR_DIR, "guard.disabled")
CLAUDE_SETTINGS = os.path.join(HOME, ".claude", "settings.json")
CLAUDE_JSON = os.path.join(HOME, ".claude.json")
GATEWAY = "http://127.0.0.1:3456"

# Env keys that mean "this config has been hijacked to point at the router".
# These belong ONLY in the repo's .claude/settings.local.json, never globally.
ROUTER_ENV_KEYS = [
    "ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE_URL", "CLAUDE_AGENT_API_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
]


def log(msg):
    print("[provision] " + msg, flush=True)


def warn(msg):
    print("[provision] WARNING: " + msg, flush=True)


def run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit("command failed: %s\n%s%s" % (cmd, r.stdout, r.stderr))
    return r


def health_ok(timeout=3):
    try:
        with urllib.request.urlopen(GATEWAY + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ccr_stop():
    run("ccr stop", check=False)
    # ccr's own stop is best-effort; make sure nothing is left holding the port.
    run("pkill -f 'claude-code-router/dist/main/cli.js'", check=False)
    run("pkill -f gateway-bootstrap.js", check=False)
    time.sleep(1)


def ccr_start(wait=25):
    run("ccr start --no-open", check=False)
    for _ in range(wait):
        if health_ok():
            return True
        time.sleep(1)
    return False


# ------------------------------------------------------------------ preflight

def preflight(install_packages):
    if install_packages:
        log("installing ccr + claude globally (npm)")
        run("npm install -g @musistudio/claude-code-router")
        run("npm install -g @anthropic-ai/claude-code")
    missing = [b for b in ("node", "npm", "ccr", "claude") if not shutil.which(b)]
    if missing:
        raise SystemExit("missing binaries: %s (re-run with --install-packages)"
                         % ", ".join(missing))
    ver = run("node -p process.versions.node").stdout.strip()
    major = int(ver.split(".")[0])
    if major < 22:
        raise SystemExit("node >= 22 required, found %s" % ver)
    log("preflight ok (node %s)" % ver)


def ensure_db():
    """First run has no config.sqlite; ccr creates it on its first start."""
    if os.path.exists(DB):
        return
    log("no config DB yet - initialising via a throwaway ccr start")
    ccr_start(wait=20)
    ccr_stop()
    if not os.path.exists(DB):
        raise SystemExit("ccr did not create %s" % DB)


# -------------------------------------------------------------------- config

def configure_db(model, or_key):
    """Write provider + routes, and hard-assert the profile invariants."""
    con = sqlite3.connect(DB)
    try:
        row = con.execute(
            "select value_json from app_config where key='default'").fetchone()
        cfg = json.loads(row[0]) if row else {}

        cfg["Providers"] = [{
            "name": "openrouter",
            "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key": or_key,
            "models": [model],
            "transformer": {"use": [["openrouter"]]},
        }]
        route = "openrouter,%s" % model
        router = cfg.setdefault("Router", {})
        for k in ("default", "background", "think", "longContext", "webSearch"):
            router[k] = route
        cfg["preferredProvider"] = "openrouter"
        cfg.setdefault("gateway", {}).update(
            {"enabled": True, "host": "127.0.0.1", "port": 3456,
             "coreHost": "127.0.0.1", "corePort": 3457})

        # ---- THE INVARIANT -------------------------------------------------
        # Global profile takeover rewrites ~/.claude/settings.json to point at
        # 127.0.0.1:3456. When ccr then stops, Claude Code has no endpoint and
        # any agent running here goes mute mid-task. It has killed two agents
        # already. Scoping is done per-repo instead. Never enable this.
        prof = cfg.setdefault("profile", {})
        prof["enabled"] = False
        for sub in ("claudeCode", "codex"):
            if isinstance(prof.get(sub), dict):
                prof[sub]["enabled"] = False
        for p in prof.get("profiles", []):
            p["enabled"] = False
            p["scope"] = "manual"
        # --------------------------------------------------------------------

        con.execute("insert into app_config(key,value_json,updated_at) "
                    "values('default',?,?) on conflict(key) do update set "
                    "value_json=excluded.value_json, updated_at=excluded.updated_at",
                    (json.dumps(cfg), time.strftime("%Y-%m-%dT%H:%M:%SZ")))

        # Gateway key: reuse if present so re-runs don't invalidate live configs.
        got = con.execute(
            "select encrypted_key from api_keys where id='local-gateway'").fetchone()
        if got and got[0]:
            key = got[0]
            log("reusing existing local-gateway key")
        else:
            key = "sk-ccr-" + secrets.token_hex(24)
            log("generated a new local-gateway key")
        con.execute(
            "insert or replace into api_keys"
            "(id,name,encrypted_key,encryption,created_at,expires_at,limits_json) "
            "values('local-gateway','Local Gateway',?,'plain',datetime('now'),'','')",
            (key,))
        con.commit()
        con.execute("pragma wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    log("config DB written (model=%s)" % model)
    return key


# --------------------------------------------------- global-settings hygiene

def clean_global_settings():
    """~/.claude/settings.json must never carry router env. Strip it, save golden."""
    data = {}
    if os.path.exists(CLAUDE_SETTINGS):
        try:
            with open(CLAUDE_SETTINGS) as f:
                data = json.load(f)
        except Exception:
            warn("global settings.json was unparseable; rewriting it empty")
            data = {}
    env = data.get("env") or {}
    removed = [k for k in ROUTER_ENV_KEYS if k in env]
    for k in removed:
        env.pop(k)
    if env:
        data["env"] = env
    else:
        data.pop("env", None)
    os.makedirs(os.path.dirname(CLAUDE_SETTINGS), exist_ok=True)
    with open(CLAUDE_SETTINGS, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    shutil.copyfile(CLAUDE_SETTINGS, GOLDEN)   # the guard restores from this
    if removed:
        warn("stripped hijacked env from global settings.json: %s" % ", ".join(removed))
    log("global settings clean; golden copy at %s" % GOLDEN)


def purge_takeover_artifacts():
    """Remove files CCR's takeover created. `.ccr-original-missing` (0 bytes) is
    CCR's own marker meaning 'this file did not exist before I made it'."""
    tm = os.path.join(CCR_DIR, "global-profile-takeover.json")
    if os.path.exists(tm):
        os.remove(tm)
        warn("removed global-profile-takeover.json")
    for f in (os.path.join(HOME, ".codex", "config.toml"),
              os.path.join(HOME, ".codex", "ccr-model-catalog.json"),
              os.path.join(HOME, ".codex", "claude-code-router.config.toml"),
              os.path.join(HOME, ".claude", ".claude.json")):
        if os.path.exists(f + ".ccr-original-missing") and os.path.exists(f):
            os.remove(f)
            os.remove(f + ".ccr-original-missing")
            warn("removed CCR-created %s" % f)


# ------------------------------------------------------------- repo + trust

def write_repo_settings(repo, key, model, ctx):
    """Per-repo scoping. THIS is how routing is applied - never globally."""
    d = os.path.join(repo, ".claude")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "settings.local.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    env = data.setdefault("env", {})
    env.update({
        "ANTHROPIC_BASE_URL": GATEWAY,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        # Claude Code's catalog does not know third-party model ids, so without
        # this it assumes a 200k window and auto-compacts long runs far too early.
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(ctx),
    })
    # A stale API key silently beats the gateway token; they are exclusive.
    env.pop("ANTHROPIC_API_KEY", None)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(path, 0o600)
    log("wrote %s" % path)
    r = run("cd %s && git check-ignore -q .claude/settings.local.json" % repo,
            check=False)
    if r.returncode != 0:
        warn("%s is NOT gitignored - it holds the gateway key. Add "
             ".claude/settings.local.json to .gitignore." % path)


def set_trust(repo):
    """Without this the repo's .claude/settings.json permissions are ignored,
    so tool-using `claude -p` calls degrade on a fresh VM."""
    data = {}
    if os.path.exists(CLAUDE_JSON):
        try:
            with open(CLAUDE_JSON) as f:
                data = json.load(f)
        except Exception:
            warn("~/.claude.json unparseable; leaving it alone")
            return
        shutil.copyfile(CLAUDE_JSON, CLAUDE_JSON + ".pre-provision")
    projects = data.setdefault("projects", {})
    entry = projects.setdefault(repo, {})
    if entry.get("hasTrustDialogAccepted") is True:
        log("workspace already trusted")
        return
    entry["hasTrustDialogAccepted"] = True
    with open(CLAUDE_JSON, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    log("marked %s as a trusted workspace" % repo)


# ---------------------------------------------------------------------- main

def resolve_key_from_db():
    """Re-runs on an already-provisioned box shouldn't demand the key again."""
    if not os.path.exists(DB):
        return ""
    try:
        con = sqlite3.connect(DB)
        row = con.execute(
            "select value_json from app_config where key='default'").fetchone()
        con.close()
        if not row:
            return ""
        cfg = json.loads(row[0])
        for p in cfg.get("Providers", []):
            if p.get("name") == "openrouter" and p.get("api_key"):
                return p["api_key"]
    except Exception:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/mnt/workspace/AutoProbe")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--context-tokens", type=int, default=1048576)
    ap.add_argument("--install-packages", action="store_true")
    ap.add_argument("--no-start", action="store_true")
    a = ap.parse_args()

    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not or_key:
        or_key = resolve_key_from_db()
        if or_key:
            log("reusing the OpenRouter key already in the config DB")
    if not or_key:
        raise SystemExit(
            "set OPENROUTER_API_KEY (https://openrouter.ai/keys) - "
            "it is never stored in this repo")
    if not os.path.isdir(a.repo):
        raise SystemExit("repo not found: %s" % a.repo)

    preflight(a.install_packages)
    ensure_db()

    # Hold the guard off: it would otherwise see a stopped gateway mid-run and
    # race us by restarting ccr while the config DB is being rewritten.
    os.makedirs(CCR_DIR, exist_ok=True)
    with open(GUARD_PAUSE, "w") as f:
        f.write("provision_ccr.py pid %d\n" % os.getpid())
    try:
        log("stopping ccr so the config DB can be written safely")
        ccr_stop()

        key = configure_db(a.model, or_key)
        clean_global_settings()
        purge_takeover_artifacts()
        write_repo_settings(a.repo, key, a.model, a.context_tokens)
        set_trust(a.repo)

        if a.no_start:
            log("done (--no-start; gateway not started)")
            return
        log("starting gateway")
        if not ccr_start():
            raise SystemExit(
                "gateway did not become healthy on %s/health" % GATEWAY)
        log("gateway healthy at %s" % GATEWAY)
    finally:
        if os.path.exists(GUARD_PAUSE):
            os.remove(GUARD_PAUSE)
    log("DONE. Verify with: bash %s/CCR-setup-bundle/verify.sh" % a.repo)


if __name__ == "__main__":
    main()
