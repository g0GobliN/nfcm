# Inference backends

NFCM routes chat/inference through the `InferenceBackend` trait.

## Backends

| Kind | Env | Status |
|------|-----|--------|
| `mock` (default) | `NFCM_INFERENCE_BACKEND=mock` | Always available; labeled mock |
| `candle` | `NFCM_INFERENCE_BACKEND=candle` | Needs `--features candle`; scaffold only |
| `gguf` | `NFCM_INFERENCE_BACKEND=gguf` | Needs llama.cpp CLI + `.gguf` file |

## Select backend

```bash
export NFCM_INFERENCE_BACKEND=mock     # default
export NFCM_INFERENCE_BACKEND=candle   # build with --features candle
export NFCM_INFERENCE_BACKEND=gguf
export NFCM_LLAMA_CLI=llama-cli
export NFCM_GGUF_MODEL=/path/to/model.gguf
```

Or in code:

```rust
RuntimeConfig::new(dir).with_backend(BackendKind::Gguf)
```

## Candle feature

```bash
cd nfc-runtime
cargo test -p nfc-runtime --features candle
```

Default CI does **not** enable `candle` (keeps deps light).

## GGUF / llama.cpp

1. Install a llama.cpp CLI that accepts `-m`, `-p`, `-n`
2. Point `NFCM_GGUF_MODEL` at a `.gguf`
3. Start runtime with `NFCM_INFERENCE_BACKEND=gguf`
4. Load/compile a brain, then chat

If CLI or model is missing, attach fails with a clear error — no fake output.

## Honesty

- Mock responses set `is_mock: true`
- Candle scaffold banners that generative weights are not loaded yet
- GGUF output is whatever your local llama.cpp model produces (your responsibility)
