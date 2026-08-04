# NFCM Runtime

<p align="center">
  <img src="../assets/logo/logo2.png" alt="NFCM" width="128" height="128" />
</p>

Local AI runtime package for the [NFCM](../README.md) project.

> Phase 1 platform only — mock generator, no trained 32B compressor.

## Docs

Canonical docs live at the repo root:

- [Structure](../docs/STRUCTURE.md)
- [Getting started](../docs/getting-started.md)
- [Architecture](../docs/architecture.md)
- [Inference backends](../docs/inference-backends.md)
- [Research](../docs/research-notes.md)
- [Contributing](../CONTRIBUTING.md)

Package-local mirrors in [`docs/`](docs/) point to the same content.

## Quick commands

```bash
cargo test --workspace --exclude nfc-desktop
cargo run -p nfc-runtime --example smoke

cd apps/desktop && npm install && npm run tauri dev
```

## Crates

| Crate | Role |
|-------|------|
| `nfc-runtime` | Engine |
| `nfc-generator` | `WeightGenerator` + mock |
| `nfc-storage` | Registry + cache |
| `nfc-hardware` | Device probes |
| `nfc-tensor` | Tensor helpers |

## License

MIT — see [../LICENSE](../LICENSE).
