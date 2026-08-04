# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project logo / branding assets (`branding/`) + Tauri & UI icons
- Pluggable `InferenceBackend` trait (mock default, Candle feature, GGUF/llama.cpp)
- Runtime snapshot field `inference_backend` + UI surface
- Docs: `docs/inference-backends.md`
- CI: rustfmt, clippy (`-D warnings`), tests, smoke, frontend build + aggregate gate
- Workflow: reject AI co-author / Generated-by commit trailers
- Workflow: PR title + secret/large-file sanity checks
- Dependabot for Cargo, npm, and GitHub Actions
- CODEOWNERS, label catalog, maintainer docs for branch protection

### Changed

- (none yet)

### Fixed

- (none yet)

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
