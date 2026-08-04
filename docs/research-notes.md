# Research roadmap

Phase 1 built the **runtime platform**. Phase 2 adds a **latent → weights** prototype. Phase 3 starts the **skill codebook**.

## Done / in progress

1. **Latent foundation (proto)** — see [phase-2.md](phase-2.md): `LatentWeightGenerator` + `eval_generators` example.
2. **Backend adapters** — mock / Candle feature / GGUF CLI / latent probe.
3. **Persist + reload** — full `{id}.nfcm.json` artifacts with tensors.
4. **Toy hypernetwork** — `experiments/.../hypernetwork/train_toy.py` → `toy-v1.json` checkpoint.
5. **Desktop Settings** — switch backend + import GGUF path.
6. **Skill codebook (Phase 3)** — see [phase-3.md](phase-3.md): compressed skill residuals, dynamic topology, toy trainer, LRU pager.

## Near-term

1. **Trained codebook residuals** — replace hash synth with richer task-loss embeddings.
2. **Real hypernetwork losses** — replace teacher-mimic with task metrics.
3. **Dynamic subnetworks** — further specialize layers beyond depth/width scaling.
4. **Eval harness** — see [eval.md](eval.md): fixed-RAM footprint, latent discrimination, probe energy.

## Non-goals (yet)

- Claiming 32B-equivalent quality
- Cloud sync / hosted inference
- Shipping a full foundation model checkpoint as “done”

## How to contribute research

- Open an issue with the experiment hypothesis and RAM budget.
- Keep training code in `nfc-runtime/experiments/` initially.
- Do not merge unlabelled mock results as production claims.
- Document metrics, hardware, and seeds in the PR.
