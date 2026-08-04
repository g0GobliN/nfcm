# Fixed-RAM eval suite

Honest plumbing metrics for generators — **not** LLM quality claims.

## Run

```bash
cd nfc-runtime
cargo run -p nfc-runtime --example eval_generators
NFCM_EVAL_BUDGET_MB=128 cargo run -p nfc-runtime --example eval_generators
```

## What it measures

| Metric | Meaning |
|--------|---------|
| `allocated_bytes` / `claimed_mb` | Real tensor footprint vs soft budget claim |
| `under_claim` | Allocated tensors fit under the claimed memory |
| `skill_hits` / `codebook_id` | Codebook activation |
| `discrimination.separation` | Same-domain latent cosine − cross-domain (higher is better separation) |
| `probe_energy.delta` | Matched-prompt energy − mismatched (latent-probe only) |

## Library API

```rust
use nfc_generator::{create_generator, run_suite, GeneratorKind};

let gen = create_generator(GeneratorKind::Latent);
let report = run_suite(gen.as_ref(), 256 * 1024 * 1024);
```

See also [phase-3.md](phase-3.md) and [research-notes.md](research-notes.md).
