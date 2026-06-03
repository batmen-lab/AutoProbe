PYTHON := venv/bin/python
PIP    := venv/bin/pip

# claude-code-router routes the `claude` CLI through OpenRouter (config:
# ~/.claude-code-router/config.json). `make api` and `make cli` auto-start
# the router and inject its env vars (ANTHROPIC_BASE_URL etc.) before
# launching python, so any `claude` subprocess the pipeline spawns is
# transparently routed. To swap the model, run:
#     ~/.claude-code-router/set-model.sh <openrouter-model-id> && ccr restart

.PHONY: setup api web cli help ccr-up ccr-down api-codex shim-up shim-down

help:
	@echo "make setup     — create venv and install backend + training deps; npm install for the web"
	@echo "make api       — start FastAPI on :8765 (claude backend, auto-starts ccr)"
	@echo "make api-codex — start FastAPI on :8766 (codex backend, uses your ChatGPT subscription via 'codex' CLI; no ccr)"
	@echo "make web       — start Next.js dev server on :3000"
	@echo "make cli       — interactive CLI (auto-starts ccr)"
	@echo "make ccr-up    — ensure claude-code-router is running"
	@echo "make ccr-down  — stop claude-code-router"
	@echo "make shim-up   — start the gemini reasoning-injector shim on :4000"
	@echo "make shim-down — stop the shim"

setup:
	test -d venv || python3 -m venv venv
	$(PIP) install -q -r requirements.txt
	cd web && npm install

ccr-up:
	@ccr status 2>/dev/null | grep -q Running || { echo "Starting ccr..."; nohup ccr start >/dev/null 2>&1 & sleep 1.5; }

ccr-down:
	-@ccr stop 2>/dev/null

# Reasoning-injector shim: sits between ccr and OpenRouter and splices in the
# `reasoning` field so gemini-flash/gemini-pro/codex-max stop 400-ing. ccr's
# config points api_base_url at http://127.0.0.1:4000/v1/chat/completions.
shim-up:
	@curl -s -o /dev/null -w "" http://127.0.0.1:4000/ 2>/dev/null && echo "shim already up" || { \
		echo "Starting gemini shim on :4000..."; \
		nohup $(PYTHON) -m tools.ccr_gemini_shim > /tmp/ccr_gemini_shim.log 2>&1 & \
		sleep 0.8; }

shim-down:
	-@fuser -k -TERM 4000/tcp 2>/dev/null

api: shim-up ccr-up
	-@fuser -k -TERM 8765/tcp 2>/dev/null; sleep 0.3
	@eval "$$(ccr activate)" && $(PYTHON) -m server.app

web:
	cd web && npm run dev

cli: shim-up ccr-up
	@eval "$$(ccr activate)" && $(PYTHON) main.py

# Codex backend — same server.app, swapped via LLM_BACKEND=codex env var.
# Runs land under response_codex/ (workspace.py routes RUN_BASE on the flag).
# No ccr involvement — codex CLI talks to OpenAI directly via your subscription.
api-codex:
	-@fuser -k -TERM 8766/tcp 2>/dev/null; sleep 0.3
	@command -v codex >/dev/null || { echo "codex CLI not found — install @openai/codex and run 'codex login'"; exit 1; }
	LLM_BACKEND=codex API_PORT=8766 $(PYTHON) -m server.app
