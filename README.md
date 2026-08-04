# NFCM

**Neural Foundation Compression Model** — a local-first AI runtime that aims to generate task-specific neural subnetworks at runtime instead of shipping one huge general LLM.

> **Honesty first:** Phase 1 is an **engineering platform** (runtime, registry, memory manager, mock generator, desktop UI). It does **not** yet implement a production 32B compression algorithm or claim trained-model quality.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-workspace-orange.svg)](nfc-runtime/Cargo.toml)
[![Status](https://img.shields.io/badge/status-phase%201%20foundation-blue.svg)](CHANGELOG.md)

---

## Vision

Long-term goal: make high-capability AI usable on low-memory devices by:

- storing **compressed knowledge representations**
- **generating** task-specific subnetworks at runtime
- activating only required capabilities
- optimizing memory dynamically

Phase 1 builds the **plug-in surface** where that research can land (`WeightGenerator` trait).

## What's in this repo

| Path | Purpose |
|------|---------|
| [`nfc-runtime/`](nfc-runtime/) | Runtime framework + Tauri desktop app |
| [`docs/`](docs/) | Project docs, structure, architecture |
| [`nfc-runtime/experiments/`](nfc-runtime/experiments/) | Research sandbox (not production) |

## Quick start

### Prerequisites

- Rust 1.75+ (1.97 tested)
- Node.js 20+
- Linux Tauri deps (see [Getting Started](docs/getting-started.md))

### Runtime smoke test

```bash
cd nfc-runtime
cargo test --workspace --exclude nfc-desktop
cargo run -p nfc-runtime --example smoke
```

### Desktop app

```bash
cd nfc-runtime/apps/desktop
npm install
npm run tauri dev
```

Frontend-only preview (no Rust shell):

```bash
npm run dev
```

Data dir: `~/.local/share/nfcm/`

## Documentation

| Doc | Description |
|-----|-------------|
| [Structure](docs/STRUCTURE.md) | Full repository layout |
| [Getting started](docs/getting-started.md) | Install & run |
| [Architecture](docs/architecture.md) | Runtime design |
| [Research roadmap](docs/research-notes.md) | Next scientific steps |
| [Maintainer / CI](docs/maintainer.md) | Branch protection, labels, Dependabot |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Code of Conduct](CODE_OF_CONDUCT.md) | Community norms |
| [Security](SECURITY.md) | Vulnerability reporting |
| [Changelog](CHANGELOG.md) | Release history |
| [Support](SUPPORT.md) | Where to get help |

## Phase 1 status

| Component | Status |
|-----------|--------|
| Cargo workspace | Done |
| Hardware detection | Done |
| Model registry (SQLite) | Done |
| Memory manager | Done |
| Mock `WeightGenerator` | Done |
| Task compiler (heuristic) | Done |
| Runtime engine | Done |
| Tauri + React UI | Done |
| Real hypernetwork | Not started |
| Candle / ONNX / llama.cpp | Future |

## Project principles

1. **No fake AI claims** — mock paths are labeled (`is_mock: true`).
2. **Local-first** — no cloud dependency for core runtime.
3. **Separable layers** — framework ≠ research algorithm ≠ current mock.
4. **Low memory** — soft budgets; avoid unnecessary allocations.
5. **Clean seams** — swap generators/backends without rewriting the UI.

## Contributing

Contributions are welcome — code, docs, tests, research notes, and issue triage.

1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
3. Open an issue before large design changes
4. Keep PRs focused and tested

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for people who have helped shape the project.

## License

Released under the [MIT License](LICENSE).

## Disclaimer

NFCM Runtime Phase 1 is research/infrastructure software. Generated “brains” from the mock generator are **placeholders**, not production models. Do not use mock inference for medical, legal, or safety-critical decisions.
