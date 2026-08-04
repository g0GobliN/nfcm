#!/usr/bin/env bash
# Fixed-RAM generator eval (structural metrics only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export NFCM_EVAL_BUDGET_MB="${NFCM_EVAL_BUDGET_MB:-256}"
echo "NFCM_EVAL_BUDGET_MB=$NFCM_EVAL_BUDGET_MB"
exec cargo run -p nfc-runtime --example eval_generators "$@"
