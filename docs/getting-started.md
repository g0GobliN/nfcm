# Getting started

## Prerequisites

- **Rust** 1.75+ ([rustup](https://rustup.rs/))
- **Node.js** 20+ and npm
- **Linux** recommended for Tauri 2 desktop (macOS/Windows possible with platform deps)

### Debian / Ubuntu Tauri deps

```bash
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  libjavascriptcoregtk-4.1-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  build-essential \
  curl wget file \
  libssl-dev \
  libxdo-dev \
  pkg-config
```

## Clone

```bash
git clone <this-repo-url>
cd nfcm
```

## Test the Rust workspace

```bash
cd nfc-runtime
cargo test --workspace --exclude nfc-desktop
cargo run -p nfc-runtime --example smoke
```

## Run the desktop app

```bash
cd nfc-runtime/apps/desktop
npm install
npm run tauri dev
```

### Frontend-only preview

Useful for UI work without WebKit/GTK:

```bash
cd nfc-runtime/apps/desktop
npm install
npm run dev
```

The UI shows a labeled preview snapshot when Tauri is not connected.

## Data directory

Runtime state is local:

```
~/.local/share/nfcm/
```

## Next

- [Architecture](architecture.md)
- [Structure](STRUCTURE.md)
- [Contributing](../CONTRIBUTING.md)
