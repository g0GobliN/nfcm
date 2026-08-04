# NFCM research method (yours)

**Name:** Task-Activated Residual Compaction (**TARC**)

**Claim (honest):** a *platform + training recipe* for compiling RAM-bounded specialists from a compressed skill bank — not a finished foundation model.

## Loop

```text
1. Compress skills → codebook residuals (task-loss trained)
2. Activate only needed skills for a TaskProfile
3. Blend residuals into a latent z
4. Hypernet decode z → tiny weight tensors (task-metric trained)
5. Page cold skills under a RAM budget
6. Forward pass + **greedy text decode** (fixed vocab — not an LLM)
7. Score at fixed RAM (footprint + discrimination + probe energy)
```

This loop is implemented in-repo:

| Piece | Path |
|-------|------|
| Codebook task-loss trainer | `codebook/train_task_loss.py` |
| Hypernet task-metric trainer | `hypernetwork/train_task.py` |
| Runtime pager + generator | `nfc-generator` (Rust) |
| Fixed-RAM eval | `docs/eval.md` / `scripts/eval-suite.sh` |

## What you can say is yours

- The **TARC** loop and seams (activate → latent → decode → page → eval)
- These **training recipes** and toy checkpoints under `experiments/`
- The **eval protocol** (under-claim, separation, probe delta)

## What you cannot claim yet

- 32B-class quality
- Beating a real GGUF chat model on language tasks
- A production foundation checkpoint

When a checkpoint *does* beat a small open GGUF on your private eval **and** you want it as product IP → keep that weight file private (engine stays open).

## Reproduce

```bash
cd nfc-runtime/experiments/neural-generation/codebook
python3 train_task_loss.py

cd ../hypernetwork
python3 train_task.py

cd ../../..
./scripts/eval-suite.sh
```
