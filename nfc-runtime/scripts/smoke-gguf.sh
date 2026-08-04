#!/usr/bin/env bash
# Headless real inference smoke (TinyLlama).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export NFCM_LLAMA_CLI="${NFCM_LLAMA_CLI:-$ROOT/tools/llama-b10250/llama-completion}"
if [[ -z "${NFCM_GGUF_MODEL:-}" ]]; then
  if [[ -f "$ROOT/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" ]]; then
    export NFCM_GGUF_MODEL="$ROOT/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
  else
    export NFCM_GGUF_MODEL="$ROOT/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  fi
fi
export NFCM_INFERENCE_BACKEND=gguf
export NFCM_WEIGHT_GENERATOR=mock

cd "$ROOT"
cargo run -p nfc-runtime --example smoke_gguf
