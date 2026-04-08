# Healthcare QA Chatbot — developer convenience targets
# Usage: make <target>
#
# Requires: python3, pip, venv at ./venv (or activated virtualenv)

PYTHON    := python3
VENV      := venv
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
UVICORN   := $(VENV)/bin/uvicorn
STREAMLIT := $(VENV)/bin/streamlit

.PHONY: help install test test-all run run-frontend start stop eval eval-quick eval-table \
        lint fmt clean docker-build docker-run docker-up docker-down docker-logs freeze paper

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  install       — create venv and install all dependencies"
	@echo "  test          — run fast test suite (excludes slow/GPU tests)"
	@echo "  test-all      — run full test suite"
	@echo "  run           — start FastAPI backend (port 8000)"
	@echo "  run-frontend  — start Streamlit frontend (port 8501)"
	@echo "  start         — start both API + frontend (background)"
	@echo "  stop          — stop all background services"
	@echo "  eval          — run paper evaluation on 97-question test set"
	@echo "  eval-quick    — quick eval on 10 questions (smoke test)"
	@echo "  eval-table    — print LaTeX comparison table"
	@echo "  freeze        — update requirements.lock from current venv"
	@echo "  paper         — compile paper/paper.tex to PDF (requires pdflatex)"
	@echo "  docker-build  — build Docker image"
	@echo "  docker-run    — run Docker image (CPU, port 8000)"
	@echo "  docker-up     — docker-compose up -d (API + frontend)"
	@echo "  docker-down   — docker-compose down"
	@echo "  docker-logs   — tail docker-compose logs"
	@echo "  clean         — remove __pycache__, .pyc files"

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -q -m "not slow" --tb=short

test-all:
	$(PYTEST) tests/ -v --tb=short

# ── Run ───────────────────────────────────────────────────────────────────────
run:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
	HF_HOME=$(HOME)/.cache/huggingface \
	$(PYTHON) api/main.py

run-frontend:
	$(STREAMLIT) run frontend/streamlit_app.py --server.port 8501

start:
	./start.sh

stop:
	./start.sh --stop

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	$(PYTHON) evaluation/run_paper_eval.py --mode metrics --n 97

eval-quick:
	$(PYTHON) evaluation/run_paper_eval.py --mode metrics --n 10

eval-table:
	$(PYTHON) evaluation/generate_baseline_table.py --format markdown

# ── Reproducibility ───────────────────────────────────────────────────────────
freeze:
	$(PIP) freeze > requirements.lock
	@echo "requirements.lock updated"

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t healthcare-qa:latest .

docker-run:
	docker run --rm -p 8000:8000 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		--env-file .env \
		healthcare-qa:latest

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ── Lint / Format ─────────────────────────────────────────────────────────────
lint:
	$(VENV)/bin/ruff check src/ api/ evaluation/ tests/ --fix

fmt:
	$(VENV)/bin/ruff format src/ api/ evaluation/ tests/

# ── Paper ─────────────────────────────────────────────────────────────────────
paper:
	cd paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
	@echo "PDF: paper/paper.pdf"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned"
