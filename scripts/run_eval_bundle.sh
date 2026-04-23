#!/usr/bin/env bash
# One-shot evaluation bundle: retrieval smoke + offline e2e + manifests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"

STRICT=false
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=true
fi

RUN_ID="bundle_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="evaluation/results/${RUN_ID}"
mkdir -p "${OUT_DIR}"

RET_JSON="${OUT_DIR}/retrieval_benchmark.json"
python evaluation/run_retrieval_benchmark.py --smoke --out "${RET_JSON}"
python evaluation/run_e2e_benchmark.py --offline --run-id "${RUN_ID}"

HIT=$(jq -r '."hit@10" // 0' "${RET_JSON}")
MRR=$(jq -r '."mrr@10" // 0' "${RET_JSON}")

warn_or_fail() {
  local msg="$1"
  if [[ "${STRICT}" == "true" ]]; then
    echo "REGRESSION: ${msg}"
    exit 1
  else
    echo "WARNING: ${msg}"
  fi
}

if awk "BEGIN {exit !(${HIT} < 0.70)}"; then
  warn_or_fail "hit@10 (${HIT}) dropped below 0.70"
fi

if awk "BEGIN {exit !(${MRR} < 0.40)}"; then
  warn_or_fail "mrr@10 (${MRR}) dropped below 0.40"
fi

echo "Evaluation bundle complete. See evaluation/results/"
