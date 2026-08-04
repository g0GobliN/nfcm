# Architecture

## Layers

1. **Desktop shell** (`nfc-runtime/apps/desktop`) — Tauri commands → React UI. No cloud.
2. **Runtime** (`nfc-runtime`) — `RuntimeEngine` owns lifecycle: compile → register → load → infer → unload.
3. **Generator** (`nfc-generator`) — `WeightGenerator` trait. Phase 1: `MockWeightGenerator`.
4. **Storage** (`nfc-storage`) — SQLite registry + filesystem cache.
5. **Hardware** (`nfc-hardware`) — soft memory budgets from live RAM/CPU/GPU probes.
6. **Tensor** (`nfc-tensor`) — shape/dtype helpers; Candle can replace buffers later.

```
┌─────────────────────────────────────────┐
│  Desktop (Tauri + React)                │
└─────────────────┬───────────────────────┘
                  │ invoke commands
┌─────────────────▼───────────────────────┐
│  RuntimeEngine                          │
│  load / unload / infer / optimize       │
├─────────────┬─────────────┬─────────────┤
│ Generator   │ Storage     │ Hardware    │
│ MemoryMgr   │ Scheduler   │ Tensor      │
└─────────────┴─────────────┴─────────────┘
```

## Model ≠ weight file

`Model` carries id, name, size, architecture, task type, memory requirement, status, and skills.

Task specialists (coding/python, medical/research) are **profiles**, not separate binary formats.

## Memory model

Default soft budget ~1 GiB (or half of available RAM):

- Generator reserve ~300 MB (accounting)
- Active model (claimed footprint from generator)
- Cache reserve ~100 MB

`optimize_memory()` drops active-model allocations when over budget.

## Inference honesty

`run_inference` goes through `InferenceBackend`. Default is mock (`is_mock: true`).
Candle (optional feature) and GGUF/llama.cpp are pluggable — see [inference-backends.md](inference-backends.md).

## Future backends

`Architecture` includes `Onnx`, `Gguf`, `Candle`. Runtime backends: `BackendKind::{Mock, Candle, Gguf}`.

## Research seam

Implement `WeightGenerator::generate(TaskProfile) -> GeneratedModel` and register it in the engine. Keep experiments under `nfc-runtime/experiments/` until stable.
