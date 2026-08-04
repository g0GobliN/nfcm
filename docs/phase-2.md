# Phase 2 — Latent weight generation

## Goal

Prove the research seam: **`TaskProfile` → `LatentCode` → real (tiny) weight tensors**.

This is an **untrained deterministic prototype**, not a foundation model.

## Select generator

```bash
export NFCM_WEIGHT_GENERATOR=mock     # default (Phase 1)
export NFCM_WEIGHT_GENERATOR=latent   # Phase 2 prototype
```

Or in code: `RuntimeConfig::new(dir).with_generator(GeneratorKind::Latent)`.

## What latent-proto does

1. Encode domain / skills / language / memory band → 64-d latent  
2. Choose small topology from latent energy  
3. Synthesize f32 matrices with a latent outer-product recipe (+ tiny noise)  
4. Report a **claimed** memory footprint for the memory manager (budget demo)  
5. Keep **allocated** tensors ≤ ~4 MiB so laptops stay happy  

`GeneratedModel.is_mock = false` for latent, but notes always say **untrained**.

## Eval harness

```bash
cd nfc-runtime
cargo run -p nfc-runtime --example eval_generators
```

Compares mock vs latent on structural metrics only (layers, latent dim, claimed vs allocated bytes). No quality claims.

## Next research steps

- Expand toy hypernet beyond teacher-mimic (real task losses)  
- Persist a larger codebook under `experiments/`  
- Score specialists at fixed RAM with a real eval suite  
- Swap latent-probe for a trained decode / Candle path  

## Inference seam (Phase 2.5)

With `NFCM_WEIGHT_GENERATOR=latent`, inference auto-selects `latent-probe-v1` (unless `NFCM_INFERENCE_BACKEND` is set). Chat runs a tiny CPU forward on the generated tensors and returns skill affinity / energy — **not** LLM text.

Compiled brains are saved as `{id}.nfcm.json` (full tensors) and reload via Model Manager.

## Trained toy hypernetwork

```bash
cd nfc-runtime/experiments/neural-generation/hypernetwork
python3 train_toy.py
```

See [inference-backends.md](inference-backends.md) and [research-notes.md](research-notes.md).
