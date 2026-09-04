#!/usr/bin/env bash
# Provision CCR across several AutoProbe worker VMs over SSH.
#
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   bash fleet_install.sh vm-1 vm-2 vm-3 vm-4
#
# Each host must already have the AutoProbe checkout at $REPO and be reachable
# by that SSH alias. Each VM gets its OWN gateway key - nothing is cloned.
#
# Env: REPO, MODEL, CONTEXT_TOKENS, INSTALL_PACKAGES=1 (bare VMs), BUNDLE_SRC
set -uo pipefail

REPO="${REPO:-/mnt/workspace/AutoProbe}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash-0731}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-1048576}"
INSTALL_PACKAGES="${INSTALL_PACKAGES:-0}"
BUNDLE_SRC="${BUNDLE_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ $# -eq 0 ]; then
  echo "usage: OPENROUTER_API_KEY=... bash fleet_install.sh <host> [host...]" >&2
  exit 2
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set" >&2
  exit 2
fi

OK=(); BAD=()
for host in "$@"; do
  printf '\n############ %s ############\n' "$host"

  # Ship the bundle itself, so a host whose checkout predates it still works.
  if ! ssh -o ConnectTimeout=20 -o BatchMode=yes "$host" \
        "mkdir -p '$REPO/CCR-setup-bundle'" 2>/dev/null; then
    echo "  cannot ssh or repo missing at $REPO"; BAD+=("$host"); continue
  fi
  scp -q -o BatchMode=yes -r \
      "$BUNDLE_SRC/." "$host:$REPO/CCR-setup-bundle/" || {
    echo "  scp failed"; BAD+=("$host"); continue; }

  # The key travels through the environment, never written to a file we ship.
  if ssh -o BatchMode=yes "$host" \
       "OPENROUTER_API_KEY='$OPENROUTER_API_KEY' REPO='$REPO' MODEL='$MODEL' \
        CONTEXT_TOKENS='$CONTEXT_TOKENS' INSTALL_PACKAGES='$INSTALL_PACKAGES' \
        bash '$REPO/CCR-setup-bundle/install.sh'"; then
    OK+=("$host")
  else
    BAD+=("$host")
  fi
done

printf '\n================ fleet summary ================\n'
printf 'ok     (%d): %s\n' "${#OK[@]}" "${OK[*]:-none}"
printf 'failed (%d): %s\n' "${#BAD[@]}" "${BAD[*]:-none}"
[ "${#BAD[@]}" -eq 0 ] || exit 1
