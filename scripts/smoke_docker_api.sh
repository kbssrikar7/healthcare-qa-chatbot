#!/usr/bin/env bash
# Smoke-test a running API (local or Docker-mapped port).
# Usage: ./scripts/smoke_docker_api.sh [BASE_URL]
# Example: ./scripts/smoke_docker_api.sh http://127.0.0.1:8000
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)"

python3 - <<PY
import json
import os
import sys
import urllib.error
import urllib.request

base = "${BASE}".rstrip("/")
headers = {"Content-Type": "application/json"}
key = os.environ.get("API_KEY", "")
if key:
    headers["X-API-Key"] = key

def get(path):
    req = urllib.request.Request(base + path, headers=headers if path.startswith("/ask") else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

def post_json(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())

try:
    h = get("/health")
except urllib.error.HTTPError as e:
    print("GET /health failed:", e.code, e.read().decode()[:500], file=sys.stderr)
    sys.exit(1)
except urllib.error.URLError as e:
    print("GET /health failed:", e, file=sys.stderr)
    sys.exit(1)

assert h.get("status") == "healthy", h

try:
    a = post_json("/ask", {"question": "What is hypertension in one sentence?"})
except urllib.error.HTTPError as e:
    print("POST /ask failed:", e.code, e.read().decode()[:500], file=sys.stderr)
    sys.exit(1)

for k in ("answer", "num_sources_requested", "num_sources_returned"):
    assert k in a, f"missing {k}: {a.keys()}"
print("smoke_docker_api: OK", base)
PY
