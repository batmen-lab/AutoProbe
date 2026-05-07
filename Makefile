PYTHON := venv/bin/python
PIP    := venv/bin/pip

.PHONY: setup api web cli help

help:
	@echo "make setup   — create venv and install backend + training deps; npm install for the web"
	@echo "make api     — start FastAPI on :8765"
	@echo "make web     — start Next.js dev server on :3000"
	@echo "make cli     — interactive CLI (same pipeline)"

setup:
	test -d venv || python3 -m venv venv
	$(PIP) install -q -r requirements.txt
	cd web && npm install

api:
	$(PYTHON) -m server.app

web:
	cd web && npm run dev

cli:
	$(PYTHON) main.py
