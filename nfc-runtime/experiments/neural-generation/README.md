# Neural generation experiments

Training / eval for weight generators. **Not** production LLM training.

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
