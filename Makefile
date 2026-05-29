PYTHON := venv/bin/python
PIP    := venv/bin/pip

# claude-code-router routes the `claude` CLI through OpenRouter (config:
# ~/.claude-code-router/config.json). `make api` and `make cli` auto-start
# the router and inject its env vars (ANTHROPIC_BASE_URL etc.) before
# launching python, so any `claude` subprocess the pipeline spawns is
# transparently routed. To swap the model, run:
#     ~/.claude-code-router/set-model.sh <openrouter-model-id> && ccr restart

.PHONY: setup api web cli help ccr-up ccr-down

help:
	@echo "make setup    — create venv and install backend + training deps; npm install for the web"
	@echo "make api      — start FastAPI on :8765 (auto-starts ccr)"
	@echo "make web      — start Next.js dev server on :3000"
	@echo "make cli      — interactive CLI (auto-starts ccr)"
	@echo "make ccr-up   — ensure claude-code-router is running"
	@echo "make ccr-down — stop claude-code-router"

setup:
	test -d venv || python3 -m venv venv
	$(PIP) install -q -r requirements.txt
	cd web && npm install

ccr-up:
	@ccr status 2>/dev/null | grep -q Running || { echo "Starting ccr..."; nohup ccr start >/dev/null 2>&1 & sleep 1.5; }

ccr-down:
	-@ccr stop 2>/dev/null

api: ccr-up
	-@fuser -k -TERM 8765/tcp 2>/dev/null; sleep 0.3
	@eval "$$(ccr activate)" && $(PYTHON) -m server.app

web:
	cd web && npm run dev

cli: ccr-up
	@eval "$$(ccr activate)" && $(PYTHON) main.py
