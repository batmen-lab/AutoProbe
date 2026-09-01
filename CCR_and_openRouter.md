# claude-code-router + OpenRouter — wiring AutoProbe to a cheap model

**Audience:** a fresh Claude agent setting this up on a new VM, with no memory of
how it was done before.

**What it buys you:** AutoProbe spawns a `claude` CLI subprocess for *every* NLP
and agent call, and a full pipeline run makes a lot of them. claude-code-router
("ccr") runs a local gateway that speaks the Anthropic Messages API and forwards
to a cheaper provider — here OpenRouter → DeepSeek. AutoProbe needs no code
changes to use it; `pipeline/router.py` injects the wiring into each subprocess.

It is **optional**. With `AUTOPROBE_ROUTER=off` the CLI talks to Anthropic
directly using whatever auth it normally has.

---

## 0. Commands that do not exist

This project uses **ccr v3**. Much of what is written about claude-code-router
online describes v1, and these commands will fail confusingly — the first one
silently, because it is parsed as a *profile name*:

```
ccr status      →  Profile "status" was not found or is disabled.
ccr restart     →  not a command (use: ccr stop && ccr start)
ccr activate    →  not a command (pipeline/router.py does this job)
ccr code        →  not a command (use: ccr <profile-name> [cli|app])
```

There is no `config.json` — config lives in `config.sqlite` (§2.3). If you find
yourself writing `eval "$(ccr activate)"`, stop: that is exactly what
`pipeline/router.py` replaces.

The real command surface:

```
ccr start [--host H] [--port P] [--open|--no-open] [--gateway|--no-gateway]
ccr ui | serve | web        # web = alias for serve
ccr stop
ccr <profile-name-or-id> [cli|app] [-- <agent args>]
```

## 1. Architecture

```
pipeline/llm.py                    spawns `claude -p --model opus ...`
      │                            with env from pipeline/router.py
      ▼
claude CLI                         resolves the alias `opus` locally via
      │                            ANTHROPIC_DEFAULT_OPUS_MODEL
      ▼
ccr gateway  127.0.0.1:3456        POST /v1/messages   (Anthropic wire format)
      │                            auth: x-api-key = the gateway key
      ▼
OpenRouter                         https://openrouter.ai/api/v1/chat/completions
      ▼
deepseek/deepseek-v4-flash-0731    (or whatever you configure)
```

Two ports, don't confuse them:

- **3456** — the *gateway*. Inference only: `/v1/messages`, `/v1/messages/count_tokens`,
  `/health`, `/models`. This is what `ANTHROPIC_BASE_URL` points at.
- **3458** — the *web UI* (from `ccr ui` / `ccr serve`). Browser app for editing
  config. Its URL in `service.json` carries a one-time `ccr_web_token`.

### ⚠️ The gotcha that will cost you an hour

**The v3 gateway does not alias `claude-*` model names onto your route.** Send it
`claude-opus-4-20250514` and it returns HTTP 400:

```json
{"error":{"message":"All target providers failed.","attempts":[{"stage":"model_resolution",
"message":"Model \"claude-opus-4-20250514\" is not configured for target provider openai.
Allowed models: deepseek/deepseek-v4-flash-0731."}]}}
```

The request must name a model the provider actually lists. This is why
`router.py` sets `ANTHROPIC_DEFAULT_OPUS_MODEL` etc. — `pipeline/llm.py` asks for
`--model opus`, and the CLI substitutes the real model id *before the request
leaves the box*. Never work around this by hardcoding a model in `llm.py`.

---

## 2. Setup on a fresh VM

### 2.1 Install

```bash
npm install -g @musistudio/claude-code-router     # needs node >= 22
npm install -g @anthropic-ai/claude-code          # the CLI itself
```

### 2.2 Initialize the config DB

```bash
ccr start --no-open      # creates ~/.claude-code-router/ and config.sqlite
ccr stop
```

### 2.3 Configure — two options

**Option A: the web UI** (needs a browser; usually impractical for an agent)

```bash
ccr ui                   # then open the URL, incl. its ccr_web_token
```

Add a provider, set the **default** and **background** routes, create a gateway key.

**Option B: write the SQLite directly** — headless, deterministic, and the one to
use if you have no browser. *(The round-trip below was verified against a real
v3.0.22 DB.)*

Stop ccr first (`ccr stop`), then:

```python
import sqlite3, json, os
db = os.path.expanduser("~/.claude-code-router/config.sqlite")
con = sqlite3.connect(db)
cfg = json.loads(con.execute(
    "select value_json from app_config where key='default'").fetchone()[0])

cfg["Providers"] = [{
    "name": "openrouter",
    "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
    "api_key": "sk-or-v1-YOUR_OPENROUTER_KEY",     # <-- real key here
    "models": ["deepseek/deepseek-v4-flash-0731"],  # must list every model you route to
    "transformer": {"use": [["openrouter"]]},
}]
ROUTE = "openrouter,deepseek/deepseek-v4-flash-0731"   # "<provider>,<model>"
cfg["Router"].update({k: ROUTE for k in
    ("default", "background", "think", "longContext", "webSearch")})
cfg["preferredProvider"] = "openrouter"

con.execute("update app_config set value_json=? where key='default'",
            (json.dumps(cfg),))
con.execute("insert or replace into api_keys"
            "(id,name,encrypted_key,encryption,created_at) "
            "values('local-gateway','Local Gateway',?,'plain',datetime('now'))",
            ("sk-ccr-" + os.urandom(24).hex(),))     # the gateway key
con.commit(); con.close()
```

Then `ccr start --no-open`.

Schema notes:

- `app_config` — `(key, value_json, updated_at)`; the whole config is **one JSON
  blob** under `key='default'`.
- `api_keys` — `(id, name, encrypted_key, encryption, created_at, expires_at,
  limits_json)`. `encryption='plain'` means `encrypted_key` is the literal token.
- The DB is **WAL mode**. To read it safely while ccr is running, copy
  `config.sqlite`, `-wal` and `-shm` to a temp dir and read the copy — that is
  what `pipeline/router.py` does.

---

## 3. How AutoProbe consumes it

`pipeline/router.py` reads the config DB and returns an environment. It is called
by `pipeline/llm.py` for both `Popen` sites, and by `test.py`.

| Variable | Source in the config |
|---|---|
| `ANTHROPIC_BASE_URL` | `http://{gateway.host}:{gateway.port}` |
| `ANTHROPIC_AUTH_TOKEN` | `api_keys.encrypted_key` |
| `ANTHROPIC_MODEL` | `Router.default`, provider prefix stripped |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | `Router.think` |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | `Router.default` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` | `Router.background` |

Rules it follows:

- **Your exports win.** Values already in the environment are never overwritten
  (`setdefault`), so you can override any single field by exporting it.
- `ANTHROPIC_API_KEY` is *removed* when a gateway token is set — the two are
  mutually exclusive for the CLI, and a stale key silently beats the token.
- `AUTOPROBE_ROUTER=off` disables injection entirely.
- If the DB is missing or unreadable, it degrades to a plain pass-through
  environment instead of failing the run.

There is also a `.claude/settings.local.json` in the repo carrying the same
values for *interactive* `claude` sessions started in this directory. It is
**gitignored**, so a fresh clone will not have it, and its values are a hardcoded
snapshot that can drift from the DB. The pipeline does not depend on it.

---

## 4. Verify

```bash
make doctor                          # python / node / claude / ccr / routed models
venv/bin/python -m pipeline.router   # one-line summary of resolved routing
make ccr-up                          # ensure the gateway is serving
```

Expected `router.py` output:

```
router: ccr http://127.0.0.1:3456 default=deepseek/deepseek-v4-flash-0731 small=deepseek/...
```

Direct gateway check — **use the real model id, not a `claude-*` name**:

```bash
curl -s http://127.0.0.1:3456/health
curl -s -X POST http://127.0.0.1:3456/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $(venv/bin/python -c "from pipeline import router; print(router.gateway_env()['ANTHROPIC_AUTH_TOKEN'])")" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"deepseek/deepseek-v4-flash-0731","max_tokens":64,
       "messages":[{"role":"user","content":"Reply with exactly: PONG"}]}'
```

End-to-end through the CLI:

```bash
venv/bin/python test.py
```

The **NLP** and **agent** checks must pass. The **web-search** check is
informational and is *expected to fail* when routed: `WebSearch` is Anthropic's
server-side search and returns nothing through a third-party provider. That is
why `pipeline/llm.py` keeps it out of `NLP_TOOLS` and relies on `WebFetch` plus
the local `Grep`/`Glob`/`Read` tools instead.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Profile "status" was not found or is disabled.` | You ran `ccr status`. It doesn't exist in v3. Probe `/health` instead — `make ccr-up` does. |
| `Model "claude-…" is not configured for target provider` | A `claude-*` name reached the gateway. Check `make doctor` shows a routed model; don't export `ANTHROPIC_MODEL` by hand. |
| `router: UNAVAILABLE — …` | No config DB, no gateway key, or no default route. Re-run §2.3. |
| 401 from `/v1/messages` | Missing/wrong gateway key. It's in the `api_keys` table. |
| Gateway won't bind | Something already on 3456, or a stale process. `ccr stop`, check `ss -lntp \| grep 3456`, then `ccr start --no-open`. |
| Calls still hit Anthropic | `ANTHROPIC_BASE_URL` was already exported (yours wins), or `AUTOPROBE_ROUTER=off` is set. |
| Config edits don't take effect | You wrote the DB while ccr was running. Stop it, write, start. Also copy the `-wal` file when reading. |

Don't trust a PID file to decide whether ccr is up — after a host restart a
recycled PID reads as "running" while nothing is bound. Probe the port.

---

## 6. Migrating an existing setup to a new VM

Carry these across (all live on the **boot disk**, under `$HOME`):

| Path | Why |
|---|---|
| `~/.claude-code-router/config.sqlite` **+ `-wal` + `-shm`** | The whole config: provider, OpenRouter API key, routes, gateway key. Copy the WAL files or you lose recent writes. |
| `~/.claude/.credentials.json` | Claude CLI auth, if you use it |
| `<repo>/.claude/settings.local.json` | Gitignored; a clone won't bring it |

Do **not** carry `~/.claude-code-router/service.json` — it holds a runtime PID
and web token, regenerated on every start.

Reinstall rather than copy: the ccr program itself
(`/usr/local/lib/node_modules/@musistudio/claude-code-router`) and the `claude`
CLI. `npm i -g` both.

⚠️ **A disk snapshot carries your OpenRouter API key, the ccr gateway token and
your Claude credentials in cleartext.** Treat any image made from it as secret.

---

## 7. Changing the model

Edit `Router.default` (and `background` / `think` if you want them to differ) in
the `app_config` JSON, **and make sure the model id is listed in that provider's
`models` array** — the gateway rejects anything not listed. Then `ccr stop &&
ccr start --no-open`. Nothing in AutoProbe needs to change; `router.py` picks up
the new values on next call.
