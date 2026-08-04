# Contributing to NFCM

Thanks for helping. This project is early — clear, honest contributions matter more than big unfinished features.

## Before you start

1. Read the [Code of Conduct](CODE_OF_CONDUCT.md).
2. Skim [docs/STRUCTURE.md](docs/STRUCTURE.md) and [docs/architecture.md](docs/architecture.md).
3. Search existing issues / PRs to avoid duplicates.
4. For design or research changes, **open an issue first**.

## Ways to contribute

- Bug reports and reproducible minimal cases
- Documentation and examples
- Tests for runtime / generator / storage
- Desktop UI polish (accessibility, clarity, no fake AI marketing)
- Research prototypes under `nfc-runtime/experiments/`
- Backend adapters (Candle, ONNX, GGUF) behind clear traits

## Development setup

See [docs/getting-started.md](docs/getting-started.md).

```bash
cd nfc-runtime
cargo test --workspace --exclude nfc-desktop
cargo run -p nfc-runtime --example smoke
```

Desktop:

```bash
cd nfc-runtime/apps/desktop
npm install
npm run tauri dev
```

## Coding guidelines

- Prefer small, focused PRs.
- Match existing style (Rust 2021, React + TypeScript).
- No drive-by dependency bloat.
- **Label mocks clearly** — never present placeholder inference as a trained model.
- Keep the stack local-first; do not add required cloud calls to core paths.
- Add / update tests when changing runtime behavior.
- Update docs when you change public APIs or layout.

## Commit messages

Use clear, imperative subjects:

```
Add cache eviction unit test
Fix memory budget underflow on unload
Docs: clarify WeightGenerator seam
```

### Commit attribution (required)

- Commits must be attributed to **human contributors** only.
- **Do not** add AI tools as co-authors. CI rejects trailers such as:
  - `Co-authored-by: Cursor …`
  - `Co-authored-by: Copilot …`
  - `Co-authored-by: ChatGPT …` / Claude / Gemini / etc.
  - `Generated-by: …` / `Made-with: Cursor`
- Using AI assistants to help write code is allowed; claiming them as co-authors is not.
- Sign-off / real human co-authors (people) are fine.

## Pull request checklist

- [ ] Tests pass (`cargo test --workspace --exclude nfc-desktop`)
- [ ] `cargo fmt` and `cargo clippy` clean (CI enforces)
- [ ] Frontend builds if UI changed (`npm run build` in `apps/desktop`)
- [ ] Docs updated if needed
- [ ] No secrets / large binaries committed
- [ ] Mock / research paths remain honestly labeled
- [ ] No AI co-author / generated-by trailers in commits

## CI

Every PR must pass:

| Workflow | What it checks |
|----------|----------------|
| **CI** | `rustfmt`, `clippy`, tests, smoke example, frontend build |
| **No AI co-author** | Commit trailers / AI attribution markers |
| **PR checks** | Title hygiene, secret/large-file guards |

See [docs/maintainer.md](docs/maintainer.md) for branch protection setup.
## Issue labels (suggested)

| Label | Use |
|-------|-----|
| `bug` | Something broken |
| `enhancement` | Feature request |
| `docs` | Documentation |
| `good first issue` | Friendly starter |
| `research` | Algorithm / experiment |
| `help wanted` | Maintainers want help |

## Security

Do not file security bugs as public issues. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
