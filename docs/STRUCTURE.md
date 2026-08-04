# Repository structure

This document describes the layout of the **NFCM** monorepo.

```
nfcm/
├── README.md                 # Project home (start here)
├── LICENSE                   # MIT
├── CONTRIBUTING.md
├── CONTRIBUTORS.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── SUPPORT.md
├── CHANGELOG.md
├── docs/                     # Cross-cutting documentation
│   ├── README.md             # Docs index
│   ├── STRUCTURE.md          # This file
│   ├── getting-started.md
│   ├── architecture.md
│   └── research-notes.md
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/            # CI (add as needed)
└── nfc-runtime/              # Main software package
    ├── README.md
    ├── Cargo.toml            # Rust workspace root
    ├── apps/
    │   └── desktop/          # Tauri + React + TypeScript + Tailwind
    │       ├── src/          # Frontend pages & API client
    │       └── src-tauri/    # Tauri commands → nfc-runtime
    ├── crates/
    │   ├── runtime/          # RuntimeEngine, memory, scheduler, inference
    │   ├── generator/        # WeightGenerator trait, mock, TaskCompiler
    │   ├── storage/          # Model registry (SQLite), cache
    │   ├── hardware/         # CPU / RAM / GPU detection
    │   └── tensor/           # Lightweight tensor primitives
    ├── models/               # Sample manifests / fixtures
    ├── experiments/          # Research code (not production)
    │   └── neural-generation/
    └── docs/                 # Runtime-specific notes (mirrors / links)
```

## Crates (Rust)

| Crate | Role |
|-------|------|
| `nfc-tensor` | Shape, dtype, owned buffers |
| `nfc-hardware` | Local hardware probes |
| `nfc-storage` | `Model` type, SQLite registry, cache |
| `nfc-generator` | `TaskProfile`, `WeightGenerator`, mock impl |
| `nfc-runtime` | Orchestration API used by desktop + examples |
| `nfc-desktop` | Tauri shell (`apps/desktop/src-tauri`) |

## Desktop UI pages

| Route | Page |
|-------|------|
| `/` | Dashboard |
| `/models` | Model Manager |
| `/compiler` | Task Compiler |
| `/console` | Runtime Console |
| `/chat` | Chat Playground |
| `/devtools` | Developer Tools |

## Data on disk (runtime)

Default: `~/.local/share/nfcm/`

```
~/.local/share/nfcm/
├── registry/
│   ├── registry.sqlite
│   └── models/           # *.nfcm-mock.json descriptors
└── cache/
```

## Extension points

1. **`WeightGenerator`** — replace `MockWeightGenerator` with a hypernetwork.
2. **`Architecture` enum** — wire `Candle` / `Onnx` / `Gguf` backends.
3. **`experiments/`** — train/eval without coupling to the desktop app.

## What does *not* live here (yet)

- Trained foundation / hypernetwork weights
- Cloud sync services
- Production LLM inference claiming 32B-level quality
