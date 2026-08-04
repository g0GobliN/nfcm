# Neural generation experiments

**Method:** [METHOD.md](METHOD.md) — Task-Activated Residual Compaction (**TARC**)

Training / eval for weight generators. **Not** production LLM training.

Runtime auto-loads `skill-task-v2` + `task-v2` + `decode-v2` (fallbacks v1) when present
(override with `NFCM_CODEBOOK` / `NFCM_HYPERNET_CHECKPOINT` / `NFCM_DECODE_HEAD`).

Shared toy corpus: `corpus/docs.json` (domain docs → co-occur + in-vocab decode phrases).

## Skill codebook (Phase 3 / TARC)

```bash
cd codebook
python3 train_task_loss.py    # → checkpoints/skill-task-v2.json  (preferred)
python3 train_codebook.py     # older margin-only toy
```

## Hypernetwork

```bash
cd hypernetwork
python3 train_task.py         # → checkpoints/task-v2.json  (preferred, dim=128)
python3 train_toy.py          # older teacher-mimic
```

## Eval snapshot

```bash
cd nfc-runtime
./scripts/eval-suite.sh
# results also under results/tarc-latest.json after a local capture
```

## Decode head

```bash
cd decode
python3 train_decode.py   # → checkpoints/decode-v2.json  (phrase / next-word)
```
