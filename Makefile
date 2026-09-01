PYTHON := venv/bin/python
PIP    := venv/bin/pip
UV     := $(shell command -v uv 2>/dev/null)

# Python for the venv. 3.12 — not the system 3.14 — because the SubpopBench
# case stack (torch / timm / netcal) has no 3.14 wheels yet.
VENV_PYTHON ?= 3.12

# torch wheel index. Defaults to the CPU build: this box has no GPU, and the
# PyPI default drags in ~2.5GB of unused CUDA libs. On a GPU box override it,
# e.g. `make setup PYTORCH_INDEX=https://download.pytorch.org/whl/cu126`.
PYTORCH_INDEX ?= https://download.pytorch.org/whl/cpu

# ── Model routing (claude-code-router v3) ───────────────────────────────────
# ccr routes the `claude` CLI through a cheaper provider. Config lives in
# ~/.claude-code-router/config.sqlite and the gateway speaks the Anthropic
# Messages API; `pipeline/router.py` reads that DB and injects the wiring into
# every `claude` subprocess the pipeline spawns.
#
# Manage providers, models and routes in the web UI: `make ccr-ui`.
# There is no `ccr activate` / `restart` / `status` — see CCR_and_openRouter.md.
# To bypass ccr and talk to Anthropic directly: AUTOPROBE_ROUTER=off make api

.PHONY: setup api web cli help ccr-up ccr-down ccr-ui doctor api-codex

help:
	@echo "make setup     — create the venv (python $(VENV_PYTHON)), install deps, npm install for the web"
	@echo "make api       — start FastAPI on :8765 (claude backend, auto-starts ccr)"
	@echo "make api-codex — start FastAPI on :8766 (codex backend, uses your ChatGPT subscription via 'codex' CLI; no ccr)"
	@echo "make web       — start Next.js dev server on :3000"
	@echo "make cli       — interactive CLI (auto-starts ccr)"
	@echo "make doctor    — check python / node / claude / ccr wiring"
	@echo "make ccr-up    — ensure the claude-code-router gateway is serving"
	@echo "make ccr-ui    — open the ccr web UI (providers, models, routes, keys)"
	@echo "make ccr-down  — stop claude-code-router"

setup:
ifdef UV
	test -d venv || uv venv --python $(VENV_PYTHON) venv
	VIRTUAL_ENV=$(CURDIR)/venv uv pip install --python $(PYTHON) \
		torch torchvision --index-url $(PYTORCH_INDEX)
	VIRTUAL_ENV=$(CURDIR)/venv uv pip install --python $(PYTHON) -r requirements.txt
else
	@echo "note: uv not found — falling back to python3 -m venv."
	@echo "      Make sure `python3 --version` is 3.12 or the torch wheels won't resolve."
	test -d venv || python3 -m venv venv
	$(PIP) install -q torch torchvision --index-url $(PYTORCH_INDEX)
	$(PIP) install -q -r requirements.txt
endif
	cd web && npm install

# Probe the gateway's real port (read from ccr's config DB — it is not always
# 3456) rather than trusting a pidfile: after a host restart ccr's recorded PID
# can be recycled and read as "running" while nothing is bound.
ccr-up:
	@command -v ccr >/dev/null || { echo "ERROR: ccr not installed — npm i -g @musistudio/claude-code-router"; exit 1; }
	@port=$$($(PYTHON) -m pipeline.router --port 2>/dev/null || echo 3456); \
	curl -s -o /dev/null --max-time 2 http://127.0.0.1:$$port/health && echo "ccr already serving on :$$port" || { \
		echo "ccr not serving on :$$port — starting..."; \
		nohup ccr start --no-open >/dev/null 2>&1 & \
		for i in $$(seq 1 30); do \
			curl -s -o /dev/null --max-time 2 http://127.0.0.1:$$port/health && break || sleep 0.5; \
		done; \
		curl -s -o /dev/null --max-time 2 http://127.0.0.1:$$port/health \
			&& echo "ccr is up on :$$port" \
			|| { echo "ERROR: ccr failed to bind :$$port after ~15s — run 'ccr start' by hand"; exit 1; }; \
	}

ccr-down:
	-@ccr stop 2>/dev/null

ccr-ui:
	@ccr ui

doctor:
	@echo "python   : $$($(PYTHON) --version 2>&1)  ($(PYTHON))"
	@echo "train py : $$($(PYTHON) -c 'from pipeline.stages import _train_interpreter; print(_train_interpreter())' 2>&1)"
	@echo "node     : $$(node --version 2>&1)"
	@echo "claude   : $$(claude --version 2>&1 || echo 'NOT FOUND — npm i -g @anthropic-ai/claude-code')"
	@echo "ccr      : $$(command -v ccr >/dev/null && echo installed || echo 'NOT FOUND')"
	@echo "$$($(PYTHON) -m pipeline.router 2>&1)"
	@$(PYTHON) -c "import torch, transformers, sklearn; print(f'torch {torch.__version__}  transformers {transformers.__version__}')" 2>&1

api: ccr-up
	-@fuser -k -TERM 8765/tcp 2>/dev/null; sleep 0.3
	$(PYTHON) -m server.app

web:
	cd web && npm run dev

cli: ccr-up
	$(PYTHON) main.py

# Codex backend — same server.app, swapped via LLM_BACKEND=codex env var.
# Runs land under response_codex/ (workspace.py routes RUN_BASE on the flag).
# No ccr involvement — codex CLI talks to OpenAI directly via your subscription.
api-codex:
	-@fuser -k -TERM 8766/tcp 2>/dev/null; sleep 0.3
	@command -v codex >/dev/null || { echo "codex CLI not found — install @openai/codex and run 'codex login'"; exit 1; }
	LLM_BACKEND=codex API_PORT=8766 $(PYTHON) -m server.app
