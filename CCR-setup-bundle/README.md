# CCR setup bundle — routing AutoProbe to a cheap model

**Audience:** a fresh Claude agent (or human) setting this up on a new AutoProbe
worker VM, with no memory of how it was done before.

**Read §1 before touching anything.** It describes a failure that has already
destroyed two agent sessions on two different VMs.

---

## 1. The one rule: never enable CCR global profile takeover

CCR has a feature called *profile takeover*. Enabling it makes CCR rewrite your
**global** `~/.claude/settings.json` (and `~/.codex/config.toml`) so that
`ANTHROPIC_BASE_URL` points at `http://127.0.0.1:3456`.

It looks like it works. Then CCR stops — crash, reboot, a `ccr stop`, anything —
and **Claude Code on this box has no endpoint at all**. An agent running here
goes mute mid-task and *cannot repair itself*, because the tool it would use to
repair itself is the thing that is broken. Killing CCR does not roll the files
back. This has happened twice, on `gcloud-vm` (2026-09-01) and on
`xuanhe-cksci-and-autoprobe` (2026-09-02).

**Routing is scoped per-repo instead**, via `<repo>/.claude/settings.local.json`,
which is gitignored and affects only `claude` invocations inside that directory.
This delivers the entire cost saving with none of the risk. Global takeover buys
you *nothing you do not already have*.

Concretely, in `~/.claude-code-router/config.sqlite` these must all stay false:

```
profile.enabled            profile.claudeCode.enabled     profile.codex.enabled
profile.profiles[].enabled profile.profiles[].scope == "manual"   (never "global")
```

`provision_ccr.py` re-asserts them on every run and `ccr_guard.py` repairs them
every 60 seconds. If you think you need takeover, you are about to break the box:
you don't.

---

## 2. Quick start on a fresh VM

```bash
export OPENROUTER_API_KEY=sk-or-v1-...          # https://openrouter.ai/keys
cd /mnt/workspace/AutoProbe
bash CCR-setup-bundle/install.sh                 # add INSTALL_PACKAGES=1 on a bare VM
```

That is the whole setup. It is **idempotent** — re-running it is also the repair
path, so if routing ever looks wrong, run it again before debugging by hand.

Overrides: `REPO`, `MODEL`, `CONTEXT_TOKENS`, `INSTALL_PACKAGES=1`.

For several VMs at once:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
bash CCR-setup-bundle/fleet_install.sh vm-1 vm-2 vm-3 vm-4
```

---

## 3. What is in this bundle

| File | Role |
|---|---|
| `install.sh` | One-shot setup for one VM: provision → install guard timer → verify. |
| `provision_ccr.py` | Idempotent provisioning **and** repair. Writes the config DB, asserts the §1 invariants, writes the per-repo scoping file, marks the workspace trusted. |
| `ccr_guard.py` | Runs every 60s. Undoes takeover, restarts a dead gateway. |
| `systemd/ccr-guard.{service,timer}` | Unit files (`__BUNDLE__`/`__USER__`/`__HOME__` are substituted at install). |
| `verify.sh` | Nine end-to-end checks. Non-zero exit if anything required fails. |
| `selftest_guard.py` | Deliberately breaks the box, then proves the guard repairs it. |
| `fleet_install.sh` | Runs `install.sh` across many VMs over SSH. |

Nothing here contains a secret. The OpenRouter key arrives via the environment
and is stored in `~/.claude-code-router/config.sqlite` (outside the repo); the
gateway key lands in the gitignored `.claude/settings.local.json`. **Keep it that
way — this directory is committed to git.**

---

## 4. Architecture

```
pipeline/llm.py            spawns `claude -p --model opus ...`
    |                      with env from pipeline/router.py
    v
claude CLI                 resolves the alias `opus` LOCALLY via
    |                      ANTHROPIC_DEFAULT_OPUS_MODEL
    v
ccr gateway 127.0.0.1:3456 POST /v1/messages  (Anthropic wire format)
    |                      auth: x-api-key / ANTHROPIC_AUTH_TOKEN = gateway key
    v
OpenRouter                 https://openrouter.ai/api/v1/chat/completions
    v
deepseek/deepseek-v4-flash-0731
```

Two ports, do not confuse them:

- **3456** — the *gateway*. Inference only: `/v1/messages`,
  `/v1/messages/count_tokens`, `/health` (open), `/models` (401 without a key).
  This is what `ANTHROPIC_BASE_URL` points at.
- **3458** — the *web UI* (`ccr ui` / `ccr serve`). Its URL in `service.json`
  carries a one-time `ccr_web_token`.

### The gotcha that costs an hour

**The v3 gateway does not alias `claude-*` model names onto your route.** Send it
`claude-opus-4-20250514` and you get HTTP 400:

```json
{"error":{"message":"All target providers failed.","attempts":[{"stage":"model_resolution",
"message":"Model \"claude-opus-4-20250514\" is not configured for target provider openai.
Allowed models: deepseek/deepseek-v4-flash-0731."}]}}
```

The request must name a model the provider actually lists. That is why the env
sets `ANTHROPIC_DEFAULT_OPUS_MODEL` and friends — `llm.py` asks for
`--model opus` and the CLI substitutes the real id *before the request leaves the
box*. Never work around this by hardcoding a model in `llm.py`.

---

## 5. Commands that do not exist

This is **ccr v3**. Most writing online describes v1. In v3 any unrecognised
argument is parsed as a *profile name*, so these fail confusingly rather than
usefully:

```
ccr status    ->  Profile "status" was not found or is disabled.
ccr -v        ->  Profile "-v" was not found or is disabled.
ccr restart   ->  not a command (use: ccr stop && ccr start)
ccr activate  ->  not a command (pipeline/router.py does this job)
ccr code      ->  not a command (use: ccr <profile-name> [cli|app])
```

There is no `config.json`; config is one JSON blob in SQLite. The real surface:

```
ccr start [--host H] [--port P] [--open|--no-open] [--gateway|--no-gateway]
ccr ui | serve | web        # web = alias for serve
ccr stop
ccr <profile-name-or-id> [cli|app] [-- <agent args>]
```

**Do not use a PID file to decide whether ccr is up.** After a host restart a
recycled PID reads as "running" while nothing is bound. Probe `/health`.

---

## 6. Config DB schema

`~/.claude-code-router/config.sqlite`, **WAL mode**:

- `app_config` — `(key, value_json, updated_at)`. The entire config is one JSON
  blob under `key='default'`.
- `api_keys` — `(id, name, encrypted_key, encryption, created_at, expires_at,
  limits_json)`. `encryption='plain'` means `encrypted_key` is the literal token.
  The one we use is `id='local-gateway'`.

Rules: **stop ccr before writing** the DB, and when *reading* it while ccr runs,
copy `config.sqlite` *plus* `-wal` and `-shm` to a temp path and read the copy.
`ccr_guard.py` and `verify.sh` both do this.

There is no `sqlite3` CLI on these VMs. Use Python's `sqlite3` module, or Node's
built-in `node:sqlite` (`DatabaseSync`, Node 22+).

---

## 7. The environment that gets injected

`pipeline/router.py` reads the config DB and builds this; the same values are
written to `<repo>/.claude/settings.local.json` for *interactive* sessions.

| Variable | Source |
|---|---|
| `ANTHROPIC_BASE_URL` | `http://{gateway.host}:{gateway.port}` |
| `ANTHROPIC_AUTH_TOKEN` | `api_keys.encrypted_key` where id=`local-gateway` |
| `ANTHROPIC_MODEL` | `Router.default`, provider prefix stripped |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | `Router.think` |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | `Router.default` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` | `Router.background` |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | real window of the routed model — see below |

- **Your exports win.** `router.py` uses `setdefault`, so exporting any variable
  overrides it.
- `ANTHROPIC_API_KEY` is *removed* when a gateway token is set — the two are
  mutually exclusive and a stale key silently beats the token.
- `AUTOPROBE_ROUTER=off` disables injection entirely.
- If the DB is unreadable it degrades to pass-through instead of failing the run.
  **This is why a dead gateway is dangerous: it fails silently and bills
  Anthropic at full price.** The guard exists to stop that.

### Two things a fresh VM needs that are easy to miss

**Context window.** Claude Code's model catalog does not know third-party ids, so
it assumes **200k** and auto-compacts long runs far too early. Set
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the model's real window. For
`deepseek/deepseek-v4-flash-0731` that is **1048576** — the serving provider's
cap. (OpenRouter advertises `context_length` 1310720 for the model but
`top_provider.context_length` 1048576; use the smaller or requests will error.)
Check any new model with:

```bash
curl -s https://openrouter.ai/api/v1/models \
  | python3 -c "import json,sys;m=[x for x in json.load(sys.stdin)['data'] if x['id']=='MODEL_ID'][0];print(m['context_length'], m['top_provider'])"
```

**Workspace trust.** On a fresh VM `~/.claude.json` has no
`projects["<repo>"].hasTrustDialogAccepted`, so the repo's `.claude/settings.json`
`permissions.allow` list is **ignored entirely** and tool-using `claude -p` calls
degrade. `provision_ccr.py` sets it.

---

## 8. Verify

```bash
bash CCR-setup-bundle/verify.sh
```

Checks, in order: gateway health · profile switches off · global settings clean ·
repo scoping file present and gitignored · workspace trusted · gateway answers
`PONG` with the real model id · `claude -p` routes inside the repo · guard timer
enabled and active · whether direct Anthropic auth still exists.

AutoProbe's own checks still apply:

```bash
make doctor
venv/bin/python -m pipeline.router
venv/bin/python test.py
```

In `test.py` the **NLP** and **agent** checks must pass. The **web-search** check
is informational and is *expected to fail* when routed: `WebSearch` is
Anthropic's server-side search and returns nothing through a third-party
provider. That is why `llm.py` keeps it out of `NLP_TOOLS` and uses `WebFetch`
plus local `Grep`/`Glob`/`Read` instead.

---

## 9. The guard

`ccr_guard.py`, every 60s via `ccr-guard.timer` (and 30s after boot).

```bash
systemctl status ccr-guard.timer
journalctl -t ccr-guard -n 50          # only repairs are logged here
tail -20 ~/.claude-code-router/guard.log
```

It repairs takeover (restoring `~/.claude/settings.json` from the golden copy at
`~/.claude-code-router/golden-claude-settings.json`) and restarts a dead gateway.
It **never** touches `<repo>/.claude/settings.local.json`.

To pause it for manual maintenance:

```bash
touch ~/.claude-code-router/guard.disabled     # rm to resume
```

`provision_ccr.py` holds that same pause file while it runs, so the two cannot
race. Note the guard **no-ops entirely while paused** — if you pause it and
forget, you have no safety net.

### Proving the safety net is real

Don't trust an untested watchdog. `selftest_guard.py` hijacks
`~/.claude/settings.json`, flips every profile switch on, plants the takeover
manifest, kills the gateway, then runs the guard and checks all of it came back:

```bash
python3 CCR-setup-bundle/selftest_guard.py --yes
```

It snapshots first and restores on failure, so a failed self-test still leaves a
working box. It stops the guard timer for the duration so a scheduled run can't
steal the result, and refuses to run if the guard is paused. Verified on
`xuanhe-cksci-and-autoprobe` 2026-09-04: full recovery in ~3 seconds.
Run it after provisioning a new VM, and after any `ccr` upgrade.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Profile "X" was not found or is disabled.` | You used a v1 command. See §5. Probe `/health` instead. |
| `Model "claude-…" is not configured for target provider` | A `claude-*` name reached the gateway. See §4. Don't export `ANTHROPIC_MODEL` by hand. |
| `router: UNAVAILABLE — …` | No config DB, no gateway key, or no default route. Re-run `install.sh`. |
| 401 from `/v1/messages` | Wrong gateway key. It is `api_keys` id `local-gateway`. |
| Gateway won't bind | Something already on 3456. `ccr stop`, `ss -lntp \| grep 3456`, then `ccr start --no-open`. |
| Calls still hit Anthropic | `ANTHROPIC_BASE_URL` already exported (yours wins), `AUTOPROBE_ROUTER=off`, or **the gateway is dead** — check `/health`. |
| Config edits don't take effect | You wrote the DB while ccr was running. Stop, write, start. Copy `-wal` when reading. |
| Agent went mute / can't reach Anthropic | Classic §1 takeover. `python3 CCR-setup-bundle/ccr_guard.py` from a shell fixes it immediately. |
| `unrecognized_model` in output | Cosmetic telemetry line for the session-title query. Not an error. |

---

## 11. Fleet notes (4–8 VMs)

- **Provision fresh per VM; do not clone `config.sqlite` between boxes.** Cloning
  copies one gateway key everywhere and drags credentials into every disk image.
  `fleet_install.sh` provisions each box from the single `OPENROUTER_API_KEY`,
  generating a distinct gateway key per VM.
- **Worker VMs probably do not need Anthropic credentials at all.** If everything
  routes through CCR, don't copy `~/.claude/.credentials.json` around; the blast
  radius of a leaked snapshot then shrinks to the OpenRouter key. Only copy it if
  you actually intend to run with `AUTOPROBE_ROUTER=off`.
- **A disk snapshot carries the OpenRouter key and gateway token in cleartext.**
  Treat any image made from a provisioned box as secret.
- The gateway binds `127.0.0.1` only, so each VM is self-contained — no shared
  router, no cross-VM firewall rules.
- Re-running `install.sh` on an already-set-up box is safe and is the fastest way
  to bring a drifted VM back to spec.

---

## 12. Changing the model

Edit `Router.default` (and `background` / `think` if they should differ) and make
sure the id is listed in that provider's `models` array — the gateway rejects
anything not listed. Easiest is to re-provision:

```bash
MODEL=some/other-model CONTEXT_TOKENS=<real window> bash CCR-setup-bundle/install.sh
```

Nothing in AutoProbe needs to change; `router.py` picks up the new values on the
next call.
