#!/usr/bin/env bash
# End-to-end verification of a CCR-provisioned worker VM.
# Exits non-zero if any REQUIRED check fails.
set -uo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$BUNDLE/.." && pwd)}"
GW="http://127.0.0.1:3456"
PASS=0; FAIL=0

ok()   { printf '  PASS  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  FAIL  %s\n' "$1"; FAIL=$((FAIL+1)); }
note() { printf '  ..    %s\n' "$1"; }

echo "== 1. gateway health =="
code="$(curl -s -o /dev/null -w '%{http_code}' "$GW/health" || true)"
[ "$code" = "200" ] && ok "GET /health -> 200" || bad "GET /health -> $code (gateway down?)"

echo "== 2. profile takeover is OFF =="
python3 - <<'PY'
import json, os, shutil, sqlite3, sys
db = os.path.expanduser("~/.claude-code-router/config.sqlite")
tmp = "/tmp/ccr-verify.sqlite"
for e in ("", "-wal", "-shm"):
    if os.path.exists(db + e):
        shutil.copyfile(db + e, tmp + e)
con = sqlite3.connect(tmp)
cfg = json.loads(con.execute(
    "select value_json from app_config where key='default'").fetchone()[0])
con.close()
p = cfg.get("profile", {})
bad = []
if p.get("enabled"): bad.append("profile.enabled")
for s in ("claudeCode", "codex"):
    if isinstance(p.get(s), dict) and p[s].get("enabled"): bad.append(s)
for x in p.get("profiles", []):
    if x.get("enabled"): bad.append("profiles[%s].enabled" % x.get("id"))
    if x.get("scope") == "global": bad.append("profiles[%s].scope" % x.get("id"))
print("  %s  profile switches all off" % ("PASS " if not bad else "FAIL "), end="")
print("" if not bad else "-> " + ", ".join(bad))
print("  ..    route default = %s" % cfg.get("Router", {}).get("default"))
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo "== 3. global settings.json carries NO router env =="
if python3 -c "
import json,os,sys
p=os.path.expanduser('~/.claude/settings.json')
d=json.load(open(p)) if os.path.exists(p) else {}
env=d.get('env') or {}
hit=[k for k in env if k.startswith('ANTHROPIC_') or k.startswith('CLAUDE_AGENT_')]
print(','.join(hit)); sys.exit(1 if hit else 0)"; then
  ok "~/.claude/settings.json is clean"
else
  bad "~/.claude/settings.json is HIJACKED (see keys above) - run provision_ccr.py"
fi

echo "== 4. repo scoping file =="
LOCAL="$REPO/.claude/settings.local.json"
if [ -f "$LOCAL" ]; then
  ok "$LOCAL exists"
  python3 -c "
import json
e=json.load(open('$LOCAL'))['env']
need=['ANTHROPIC_BASE_URL','ANTHROPIC_AUTH_TOKEN','ANTHROPIC_MODEL','CLAUDE_CODE_MAX_CONTEXT_TOKENS']
miss=[k for k in need if k not in e]
print('  ..    model=%s ctx=%s' % (e.get('ANTHROPIC_MODEL'), e.get('CLAUDE_CODE_MAX_CONTEXT_TOKENS')))
raise SystemExit('  FAIL  missing keys: %s' % miss if miss else 0)"
  ( cd "$REPO" && git check-ignore -q .claude/settings.local.json ) \
    && ok "it is gitignored (gateway key stays out of git)" \
    || bad "NOT gitignored - it holds the gateway key"
else
  bad "$LOCAL missing - run provision_ccr.py"
fi

echo "== 5. workspace trust =="
python3 -c "
import json,os,sys
p=os.path.expanduser('~/.claude.json')
d=json.load(open(p)) if os.path.exists(p) else {}
sys.exit(0 if d.get('projects',{}).get('$REPO',{}).get('hasTrustDialogAccepted') else 1)" \
  && ok "workspace trusted (repo permissions honoured)" \
  || bad "workspace NOT trusted - tool-using claude calls will degrade"

echo "== 6. gateway answers with the real model id =="
KEY="$(python3 -c "import json;print(json.load(open('$LOCAL'))['env']['ANTHROPIC_AUTH_TOKEN'])" 2>/dev/null || echo)"
MODEL="$(python3 -c "import json;print(json.load(open('$LOCAL'))['env']['ANTHROPIC_MODEL'])" 2>/dev/null || echo)"
if [ -n "$KEY" ] && [ -n "$MODEL" ]; then
  BODY="$(curl -s -X POST "$GW/v1/messages" \
      -H 'content-type: application/json' -H "x-api-key: $KEY" \
      -H 'anthropic-version: 2023-06-01' \
      -d "{\"model\":\"$MODEL\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: PONG\"}]}" || true)"
  echo "$BODY" | grep -q PONG && ok "POST /v1/messages -> PONG" \
    || bad "gateway did not answer: $(echo "$BODY" | head -c 300)"
else
  bad "could not read gateway key/model from $LOCAL"
fi

echo "== 7. claude CLI routes inside the repo =="
OUT="$(cd "$REPO" && timeout 120 claude -p 'Reply with exactly: ROUTED_OK' 2>&1 | tail -3)"
echo "$OUT" | grep -q ROUTED_OK && ok "claude -p in repo -> ROUTED_OK" \
  || bad "claude -p in repo failed: $(echo "$OUT" | head -c 300)"

echo "== 8. guard timer =="
systemctl is-enabled ccr-guard.timer >/dev/null 2>&1 \
  && ok "ccr-guard.timer enabled" || bad "ccr-guard.timer not enabled"
systemctl is-active ccr-guard.timer >/dev/null 2>&1 \
  && ok "ccr-guard.timer active" || bad "ccr-guard.timer not active"
note "last guard run: $(tail -1 ~/.claude-code-router/guard.log 2>/dev/null || echo '(no log yet)')"

echo
echo "== 9. informational: does this box still have direct Anthropic auth? =="
if [ -f ~/.claude/.credentials.json ]; then
  note "~/.claude/.credentials.json present - AUTOPROBE_ROUTER=off would work here"
else
  note "no Anthropic credentials - this box is CCR-only (fine, and safer for a fleet)"
fi

echo
echo "-------- $PASS passed, $FAIL failed --------"
[ "$FAIL" -eq 0 ] || exit 1
