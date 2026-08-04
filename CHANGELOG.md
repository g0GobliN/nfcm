# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 3 skill codebook — compressed skill residuals + dynamic specialist topology + toy trainer + LRU skill pager (`NFCM_CODEBOOK_RAM_BYTES`)
- Fixed-RAM eval suite (`run_suite` / `eval_generators`) — footprint, latent discrimination, probe energy
- TARC research loop — task-loss codebook + task-metric hypernet trainers (`METHOD.md`)
- Phase 2 latent weight generator + latent-probe inference + brain artifact persist/reload
- Toy hypernetwork train script + checkpoint loader (`NFCM_HYPERNET_CHECKPOINT`)
- Desktop Settings (backend switch, GGUF import) + real-chat scripts for local llama.cpp
- Project logo (`assets/logo/`, derived `branding/`) + Tauri & UI icons
- Pluggable `InferenceBackend` trait (mock default, Candle feature, GGUF/llama.cpp)
- Runtime snapshot field `inference_backend` + UI surface
- Docs: `docs/inference-backends.md`, `docs/phase-2.md`, `docs/phase-3.md`
- CI: rustfmt, clippy (`-D warnings`), tests, smoke, frontend build + aggregate gate
- Workflow: reject AI co-author / Generated-by commit trailers
- Workflow: PR title + secret/large-file sanity checks
- Dependabot for Cargo, npm, and GitHub Actions
- CODEOWNERS, label catalog, maintainer docs for branch protection

### Changed

- Desktop chat: Claude-like shimmer wait state, viewport-safe layout, IBM Plex Sans
- GGUF backend: chat-template single-turn, prefer loaded model path over stale env

### Fixed

- Tauri icons converted to RGBA (build panic)
- Frontend production CSS build (Tailwind `@apply` / config incompat)
- Inference UI freeze during long GGUF runs (spawn_blocking)

## [0.1.0] — 2026-08-04

### Added

- Rust workspace: `nfc-tensor`, `nfc-hardware`, `nfc-storage`, `nfc-generator`, `nfc-runtime`
- Mock `WeightGenerator` and heuristic `TaskCompiler`
- SQLite model registry and filesystem cache
- Memory manager with soft RAM budgets
- Tauri + React desktop workstation (Dashboard, Models, Compiler, Console, Chat, Dev Tools)
- Headless smoke example: `cargo run -p nfc-runtime --example smoke`
- Project docs: structure, architecture, getting started, research notes
- Open-source community files: contributing, code of conduct, security, support

### Notes

- Phase 1 foundation only — mock inference is explicitly labeled; no trained 32B compressor yet.
