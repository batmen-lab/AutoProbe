#!/usr/bin/env python3
"""
Prove the guard actually recovers from a global profile takeover.

This DELIBERATELY BREAKS the box the same way CCR's takeover does — hijacks
~/.claude/settings.json, flips the profile switches on in config.sqlite, plants
the takeover manifest, and kills the gateway — then runs ccr_guard.py and checks
that everything came back.

    python3 selftest_guard.py --yes

Everything is snapshotted first and restored if the guard fails, so a failed
self-test leaves the box working. Run it after provisioning a new VM, or after
upgrading ccr, to confirm the safety net is real.
"""
import argparse
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
SETTINGS = os.path.join(HOME, ".claude", "settings.json")
MANIFEST = os.path.join(CCR_DIR, "global-profile-takeover.json")
GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccr_guard.py")
GATEWAY = "http://127.0.0.1:3456"

HIJACKED_ENV = {
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    "ANTHROPIC_BASE_URL": GATEWAY,
    "ANTHROPIC_API_BASE_URL": GATEWAY,
    "CLAUDE_AGENT_API_BASE_URL": GATEWAY,
    "ANTHROPIC_FEDERATION_RULE_ID": "ccr-local",
    "ANTHROPIC_ORGANIZATION_ID": "ccr-local",
}

results = []


def check(name, passed, detail=""):
    results.append((name, passed))
    print("  %s  %s%s" % ("PASS" if passed else "FAIL", name,
                          (" -> " + detail) if detail else ""))


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def health_ok(timeout=4):
    try:
        with urllib.request.urlopen(GATEWAY + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def read_profile():
    con = sqlite3.connect(DB)
    cfg = json.loads(con.execute(
        "select value_json from app_config where key='default'").fetchone()[0])
    con.close()
    return cfg


def write_profile(cfg):
    con = sqlite3.connect(DB)
    con.execute("update app_config set value_json=? where key='default'",
                (json.dumps(cfg),))
    con.commit()
    con.execute("pragma wal_checkpoint(TRUNCATE)")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                    help="required: confirms you accept deliberate breakage")
    a = ap.parse_args()
    if not a.yes:
        raise SystemExit("refusing to run without --yes (this breaks the box "
                         "on purpose, then repairs it)")

    # The guard no-ops while paused, which would make this test silently
    # meaningless. Refuse rather than quietly delete someone's pause file.
    pause = os.path.join(CCR_DIR, "guard.disabled")
    if os.path.exists(pause):
        raise SystemExit("the guard is paused (%s exists) - it would do nothing "
                         "and this test would be meaningless. rm it first." % pause)

    # Stop the timer so a scheduled run cannot repair the damage before we do
    # and steal the result. Restarted in the finally block.
    timer_was_active = sh("systemctl is-active --quiet ccr-guard.timer").returncode == 0
    if timer_was_active:
        sh("sudo systemctl stop ccr-guard.timer")
        print("stopped ccr-guard.timer for the duration of the test")

    snap = os.path.join(CCR_DIR, "selftest-snapshot")
    os.makedirs(snap, exist_ok=True)
    shutil.copyfile(SETTINGS, os.path.join(snap, "settings.json"))
    shutil.copyfile(DB, os.path.join(snap, "config.sqlite"))
    print("snapshot -> %s" % snap)

    try:
        print("\n-- breaking the box the way CCR takeover does --")
        # 1. hijack global settings
        with open(SETTINGS) as f:
            data = json.load(f)
        data.setdefault("env", {}).update(HIJACKED_ENV)
        with open(SETTINGS, "w") as f:
            json.dump(data, f, indent=2)
        print("   hijacked ~/.claude/settings.json")

        # 2. flip the profile switches on
        cfg = read_profile()
        prof = cfg.setdefault("profile", {})
        prof["enabled"] = True
        for s in ("claudeCode", "codex"):
            if isinstance(prof.get(s), dict):
                prof[s]["enabled"] = True
        for p in prof.get("profiles", []):
            p["enabled"] = True
            p["scope"] = "global"
        write_profile(cfg)
        print("   enabled profile takeover in config.sqlite")

        # 3. plant the manifest
        with open(MANIFEST, "w") as f:
            json.dump({"profiles": [], "version": 1}, f)
        print("   planted global-profile-takeover.json")

        # 4. kill the gateway - this is the moment an agent would go mute
        sh("ccr stop")
        sh("pkill -f 'claude-code-router/dist/main/cli.js'")
        sh("pkill -f gateway-bootstrap.js")
        time.sleep(2)
        print("   killed the gateway")
        check("box is genuinely broken before repair", not health_ok(),
              "gateway down as expected")

        print("\n-- running the guard --")
        r = sh("python3 %s" % GUARD)
        print("   " + (r.stdout.strip().replace("\n", "\n   ") or "(no output)"))
        if r.returncode != 0:
            print("   guard exit=%d stderr=%s" % (r.returncode, r.stderr[:400]))

        print("\n-- did it recover? --")
        with open(SETTINGS) as f:
            env = (json.load(f).get("env") or {})
        leftover = [k for k in HIJACKED_ENV if k in env]
        check("global settings.json restored", not leftover,
              "still hijacked: %s" % leftover if leftover else "")

        prof = read_profile().get("profile", {})
        bad = []
        if prof.get("enabled"):
            bad.append("profile.enabled")
        for s in ("claudeCode", "codex"):
            if isinstance(prof.get(s), dict) and prof[s].get("enabled"):
                bad.append(s)
        for p in prof.get("profiles", []):
            if p.get("enabled") or p.get("scope") == "global":
                bad.append(str(p.get("id")))
        check("profile switches forced back off", not bad, ", ".join(bad))
        check("takeover manifest removed", not os.path.exists(MANIFEST))
        check("gateway back up", health_ok())

    finally:
        failed = [n for n, ok in results if not ok]
        if failed:
            print("\n!! guard did NOT fully recover - restoring the snapshot")
            shutil.copyfile(os.path.join(snap, "settings.json"), SETTINGS)
            shutil.copyfile(os.path.join(snap, "config.sqlite"), DB)
            sh("ccr start --no-open")
        if timer_was_active:
            sh("sudo systemctl start ccr-guard.timer")
            print("restarted ccr-guard.timer")

    print("\n-------- %d passed, %d failed --------"
          % (sum(1 for _, o in results if o), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
