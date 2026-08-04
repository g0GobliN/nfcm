# Neural generation experiments

Training / eval for weight generators. **Not** production LLM training.

## Skill codebook (Phase 3)

```text
codebook/skill-v1.json              # deterministic defaults
codebook/train_codebook.py          # toy contrastive trainer
codebook/checkpoints/skill-trained-v1.json
```

Compressed skill residuals. The latent generator activates matching skills, blends them into `LatentCode`, and **scales subnetwork depth/width** by how many skills hit.

```bash
cd nfc-runtime/experiments/neural-generation/codebook
python3 train_codebook.py
# → checkpoints/skill-trained-v1.json (auto-picked if present)

cd nfc-runtime
NFCM_WEIGHT_GENERATOR=latent cargo run -p nfc-runtime --example eval_generators
```

Override: `NFCM_CODEBOOK=/path/to.json`

See [docs/phase-3.md](../../../docs/phase-3.md).

## Hypernetwork (toy)

```bash
cd nfc-runtime/experiments/neural-generation/hypernetwork
python3 train_toy.py
# → checkpoints/toy-v1.json
```

Runtime auto-loads that checkpoint when present (or set `NFCM_HYPERNET_CHECKPOINT=/path/to.json`).

```bash
cd nfc-runtime
NFCM_WEIGHT_GENERATOR=latent cargo run -p nfc-runtime --example smoke
# generator name becomes latent-hypernet-v1 when checkpoint loads
```

Honest scope: learns a remix of the outer-product teacher. Not foundation-model quality.
