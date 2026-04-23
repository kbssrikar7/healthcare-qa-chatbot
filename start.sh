#!/usr/bin/env bash
# Healthcare QA Chatbot — local startup script
# Starts FastAPI backend + React/assistant-ui frontend in the background
# Usage: ./start.sh [--frontend-only | --api-only | --stop]

set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
API_LOG="/tmp/mediquery_api.log"
FRONTEND_LOG="/tmp/mediquery_frontend.log"
PID_FILE="/tmp/mediquery.pids"

# Resolve venv python
PYTHON="${VENV_DIR}/bin/python"
if [[ ! -f "$PYTHON" ]]; then
    if [[ -f "venv312/bin/python" ]]; then
        PYTHON="venv312/bin/python"
    else
        PYTHON="python3"
    fi
fi
UVICORN="${VENV_DIR}/bin/uvicorn"

stop_services() {
    if [[ -f "$PID_FILE" ]]; then
        echo "Stopping services..."
        while read -r pid; do
            kill "$pid" 2>/dev/null && echo "  killed $pid" || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
        echo "Done."
    else
        echo "No running services found (no PID file)."
    fi
    exit 0
}

# ── Parse args ────────────────────────────────────────────────────────────────
RUN_API=true
RUN_FRONTEND=true
case "${1:-}" in
    --stop)          stop_services ;;
    --api-only)      RUN_FRONTEND=false ;;
    --frontend-only) RUN_API=false ;;
esac

echo "═══════════════════════════════════════════"
echo "  MediQuery — Healthcare QA Chatbot"
echo "═══════════════════════════════════════════"

# ── Kill any previous instances ───────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    echo "Cleaning up previous run..."
    while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$PID_FILE"
    rm -f "$PID_FILE"
    sleep 1
fi
> "$PID_FILE"

# ── Start FastAPI backend ─────────────────────────────────────────────────────
if [[ "$RUN_API" == true ]]; then
    echo "Starting FastAPI backend on port ${API_PORT}..."
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME="${HOME}/.cache/huggingface" \
    "$UVICORN" api.main:app \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --workers 1 \
        --log-level info \
        >"$API_LOG" 2>&1 &
    API_PID=$!
    echo "$API_PID" >> "$PID_FILE"
    echo "  API PID: $API_PID  (log: $API_LOG)"

    # Wait for health check
    echo -n "  Waiting for API to be ready..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
            echo " ready!"
            break
        fi
        echo -n "."
        sleep 3
    done
fi

# ── Start React/assistant-ui frontend ─────────────────────────────────────────
if [[ "$RUN_FRONTEND" == true ]]; then
    echo "Starting React frontend on port ${FRONTEND_PORT}..."
    cd frontend-react
    NEXT_PUBLIC_API_URL="http://localhost:${API_PORT}" \
    npm run dev -- --port "$FRONTEND_PORT" \
        >"$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    cd ..
    echo "$FRONTEND_PID" >> "$PID_FILE"
    echo "  Frontend PID: $FRONTEND_PID  (log: $FRONTEND_LOG)"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Services started:"
if [[ "$RUN_API" == true ]]; then
echo "  API:      http://localhost:${API_PORT}"
echo "  API docs: http://localhost:${API_PORT}/docs"
fi
if [[ "$RUN_FRONTEND" == true ]]; then
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
fi
echo ""
echo "  Stop: ./start.sh --stop"
echo "═══════════════════════════════════════════"
