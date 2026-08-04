# Research roadmap

Phase 1 built the **runtime platform**. Phase 2 adds a **latent → weights** prototype.

## Done / in progress

1. **Latent foundation (proto)** — see [phase-2.md](phase-2.md): `LatentWeightGenerator` + `eval_generators` example.
2. **Backend adapters** — mock / Candle feature / GGUF CLI / latent probe.
3. **Persist + reload** — full `{id}.nfcm.json` artifacts with tensors.
4. **Toy hypernetwork** — `experiments/.../hypernetwork/train_toy.py` → `toy-v1.json` checkpoint.
5. **Desktop Settings** — switch backend + import GGUF path.

## Near-term

1. **Real hypernetwork losses** — replace teacher-mimic with task metrics.
2. **Compressed knowledge store** — codebook / residual vectors on disk; activate slices per task.
3. **Dynamic subnetworks** — generate only layers needed for the compiled skill set.
4. **Memory scheduler** — page latent blocks; unload cold skills under RAM pressure.
5. **Eval harness** — expand `eval_generators` with task-suite quality metrics at fixed RAM.

## Non-goals (yet)

- Claiming 32B-equivalent quality
- Cloud sync / hosted inference
- Shipping a full foundation model checkpoint as “done”

## How to contribute research

- Open an issue with the experiment hypothesis and RAM budget.
- Keep training code in `nfc-runtime/experiments/` initially.
- Do not merge unlabelled mock results as production claims.
- Document metrics, hardware, and seeds in the PR.
