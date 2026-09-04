#!/usr/bin/env bash
# Full CCR setup for one AutoProbe worker VM. Idempotent - safe to re-run.
#
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   bash CCR-setup-bundle/install.sh
#
# Env overrides: REPO, MODEL, CONTEXT_TOKENS, INSTALL_PACKAGES=1
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$BUNDLE/.." && pwd)}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash-0731}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-1048576}"
RUN_USER="$(id -un)"
RUN_HOME="$HOME"

say() { printf '\n=== %s ===\n' "$1"; }

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as the normal user (it calls sudo itself), not as root." >&2
  exit 1
fi

say "1/3  provisioning ccr  (repo=$REPO  model=$MODEL)"
EXTRA=()
[ "${INSTALL_PACKAGES:-0}" = "1" ] && EXTRA+=(--install-packages)
python3 "$BUNDLE/provision_ccr.py" \
  --repo "$REPO" --model "$MODEL" --context-tokens "$CONTEXT_TOKENS" \
  ${EXTRA+"${EXTRA[@]}"}

say "2/3  installing the guard timer"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for unit in ccr-guard.service ccr-guard.timer; do
  sed -e "s|__BUNDLE__|$BUNDLE|g" \
      -e "s|__USER__|$RUN_USER|g" \
      -e "s|__HOME__|$RUN_HOME|g" \
      "$BUNDLE/systemd/$unit" > "$tmp/$unit"
  sudo install -m 0644 "$tmp/$unit" "/etc/systemd/system/$unit"
  echo "installed /etc/systemd/system/$unit"
done
sudo systemctl daemon-reload
sudo systemctl enable --now ccr-guard.timer
sudo systemctl start ccr-guard.service || true
systemctl status ccr-guard.timer --no-pager | head -5 || true

say "3/3  verifying"
bash "$BUNDLE/verify.sh"
