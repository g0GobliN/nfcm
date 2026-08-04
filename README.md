<p align="center">
  <img src="assets/logo/logo2.png" alt="NFCM" width="200" height="200" />
</p>

<h1 align="center">NFCM</h1>

<p align="center">
  <strong>Neural Foundation Compression Model</strong><br/>
  A local AI runtime that builds the <em>right</em> model for the job — on your machine, under your RAM budget.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT" /></a>
  <a href="nfc-runtime/Cargo.toml"><img src="https://img.shields.io/badge/Rust-workspace-orange.svg" alt="Rust" /></a>
  <a href="nfc-runtime/apps/desktop"><img src="https://img.shields.io/badge/Desktop-Tauri%20%2B%20React-blue.svg" alt="Desktop" /></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome" /></a>
</p>

---

## Why NFCM?

Shipping a 30B+ general LLM to every laptop is wasteful. Most of the time you only need a **coding** brain, a **math** brain, or a **research** brain — and you need it to fit in limited RAM.

NFCM explores a different path:

1. Keep knowledge in a **compressed foundation**
2. **Compile** a task profile (“Python coding assistant, 1GB”)
3. **Generate** a specialist subnetwork at runtime
4. Run inference **locally**, with memory you can see and control

Think of it as a workstation for *neural compression + dynamic capability* — not another cloud chatbot wrapper.

---

## What you get today

The engineering **platform** is real and usable. You can clone, run tests, open the desktop app, compile a “brain,” and talk to it through the playground.

| Built | What it does |
|-------|----------------|
| **Runtime engine** (Rust) | Load / unload / infer / optimize memory |
| **Task compiler** | Natural language → `TaskProfile` (skills, domain, RAM limit) |
| **Weight generator seam** | `WeightGenerator` trait — mock today, hypernetwork tomorrow |
| **Inference backends** | Mock default; optional Candle feature; GGUF via llama.cpp |
| **Model registry** | SQLite + local filesystem cache |
| **Hardware + memory manager** | CPU/RAM/GPU probes and soft RAM budgets |
| **Desktop workstation** | Dashboard, Models, Compiler, Console, Chat, Dev Tools |

```text
“I need a Python coding assistant”
        │
        ▼
   TaskProfile  →  WeightGenerator  →  RuntimeEngine  →  Chat
```

We label mocks clearly (`is_mock: true`). No fake “32B on your phone” claims — the research that gets us there plugs into the seams we already shipped.

---

## Quick start

**Prereqs:** Rust 1.75+, Node 20+, Linux recommended for the desktop shell ([details](docs/getting-started.md)).

```bash
git clone https://github.com/g0GobliN/nfcm.git
cd nfcm/nfc-runtime

# Core runtime
cargo test --workspace --exclude nfc-desktop
cargo run -p nfc-runtime --example smoke

# Desktop app
cd apps/desktop
npm install
npm run tauri dev          # full app
# npm run dev              # UI preview without Tauri
```

Data stays local: `~/.local/share/nfcm/`

---

## Repo map

```text
nfcm/
├── nfc-runtime/     # Rust workspace + Tauri desktop app
├── docs/            # Architecture, backends, research notes
├── assets/logo/     # Brand mark
├── experiments/     # (under nfc-runtime) research sandbox
└── .github/         # CI, issue templates, Dependabot
```

| Want to… | Start here |
|----------|------------|
| Understand design | [docs/architecture.md](docs/architecture.md) |
| Plug in Candle / GGUF | [docs/inference-backends.md](docs/inference-backends.md) |
| Push research | [docs/research-notes.md](docs/research-notes.md) |
| Navigate files | [docs/STRUCTURE.md](docs/STRUCTURE.md) |

---

## Where you can help

We want builders who care about **honest local AI infrastructure**.

- **Rust** — runtime, memory scheduler, backends (Candle / ONNX / llama.cpp)
- **Research** — hypernetworks, latent codes, eval harnesses under `experiments/`
- **Desktop** — Tauri/React UX for a real AI workstation feel
- **Docs & tests** — clarity and confidence for the next contributor

Good first areas:

- Improve task-compiler heuristics
- Harden GGUF/llama.cpp integration
- Add eval scripts with fixed RAM budgets
- Polish the desktop console / playground

Read [CONTRIBUTING.md](CONTRIBUTING.md), open an issue for big ideas, keep PRs focused. CI runs fmt, clippy, tests, and frontend build on every PR.

---

## Principles

- **Local-first** — core path has no cloud dependency  
- **Honest labeling** — mock ≠ trained model  
- **Clean seams** — swap generators/backends without rewriting the UI  
- **Memory-aware** — budgets you can inspect and control  
- **Open** — MIT, PRs welcome  

---

## Status snapshot

| Area | State |
|------|--------|
| Runtime + desktop platform | Ready to hack on |
| Mock generator & inference | Working (clearly labeled) |
| Candle / GGUF adapters | Scaffold / env-driven |
| Real hypernetwork compressor | Open research — join us |

---

## License

[MIT](LICENSE) · [Security](SECURITY.md) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Changelog](CHANGELOG.md)

*NFCM is infrastructure and research software. Mock outputs are placeholders — not advice for medical, legal, or safety-critical use.*
