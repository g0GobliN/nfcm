#!/usr/bin/env bash
# Real local chat: Qwen (or TinyLlama) GGUF + llama.cpp.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="${NFCM_LLAMA_CLI:-$ROOT/tools/llama-b10250/llama-completion}"
MODEL="${NFCM_GGUF_MODEL:-}"
if [[ -z "$MODEL" ]]; then
  if [[ -f "$ROOT/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" ]]; then
    MODEL="$ROOT/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
  elif [[ -f "$ROOT/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]]; then
    MODEL="$ROOT/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  else
    echo "No GGUF in $ROOT/models — download Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    exit 1
  fi
fi

if [[ ! -x "$CLI" && ! -f "$CLI" ]]; then
  echo "Missing llama binary at $CLI"
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing GGUF at $MODEL"
  exit 1
fi

export NFCM_LLAMA_CLI="$CLI"
export NFCM_GGUF_MODEL="$MODEL"
export NFCM_INFERENCE_BACKEND=gguf
export NFCM_WEIGHT_GENERATOR="${NFCM_WEIGHT_GENERATOR:-mock}"

echo "CLI=$CLI"
echo "MODEL=$MODEL"
echo "Starting desktop (GGUF real chat)…"
cd "$ROOT/apps/desktop"
exec npm run tauri dev
