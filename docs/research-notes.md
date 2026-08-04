# Research roadmap

Phase 1–3 platform + **TARC** research loop are in-tree.

See [METHOD.md](../nfc-runtime/experiments/neural-generation/METHOD.md).

## Done

1. **Latent foundation (proto)** — [phase-2.md](phase-2.md)
2. **Backend adapters** — mock / Candle / GGUF / latent probe
3. **Persist + reload** — `{id}.nfcm.json`
4. **Skill codebook + pager** — [phase-3.md](phase-3.md)
5. **Fixed-RAM eval** — [eval.md](eval.md)
6. **TARC trainers** — `train_task_loss.py` (codebook) + `train_task.py` (hypernet)

## Next research (harder)

1. Scale residuals / hypernet beyond toy dims with real corpora  
2. Decode path that produces usable text (not just probe energy)  
3. Beat a small open GGUF on a **private** task suite at fixed RAM  

When (3) succeeds and you want product IP → **keep that checkpoint private**; keep the engine open.

## Non-goals (yet)

- Claiming 32B-equivalent quality  
- Cloud sync / hosted inference  
- Shipping a full foundation model as “done”
