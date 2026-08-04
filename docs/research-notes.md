# Research roadmap

Phase 1 built the **runtime platform**. Research plugs into `WeightGenerator`.

## Near-term

1. **Latent foundation** — small hypernetwork: `TaskProfile` / `LatentCode` → layer weights.
2. **Compressed knowledge store** — codebook / residual vectors on disk; activate slices per task.
3. **Dynamic subnetworks** — generate only layers needed for the compiled skill set.
4. **Backend adapters** — Candle → ONNX Runtime → optional llama.cpp GGUF baselines.
5. **Memory scheduler** — page latent blocks; unload cold skills under RAM pressure.
6. **Eval harness** (`experiments/`) — compare mock vs hypernetwork at fixed RAM budgets.

## Non-goals (yet)

- Claiming 32B-equivalent quality
- Cloud sync / hosted inference
- Shipping a full foundation model checkpoint as “done”

## How to contribute research

- Open an issue with the experiment hypothesis and RAM budget.
- Keep training code in `nfc-runtime/experiments/` initially.
- Do not merge unlabelled mock results as production claims.
- Document metrics, hardware, and seeds in the PR.
