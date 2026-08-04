# Neural generation experiments

**Method:** [METHOD.md](METHOD.md) — Task-Activated Residual Compaction (**TARC**)

Training / eval for weight generators. **Not** production LLM training.

## Skill codebook (Phase 3 / TARC)

```bash
cd codebook
python3 train_task_loss.py    # → checkpoints/skill-task-v1.json  (preferred)
python3 train_codebook.py     # older margin-only toy
```

## Hypernetwork

```bash
cd hypernetwork
python3 train_task.py         # → checkpoints/task-v1.json  (preferred, task metrics)
python3 train_toy.py          # older teacher-mimic
```

## Eval snapshot

```bash
cd nfc-runtime
./scripts/eval-suite.sh
# results also under results/tarc-latest.json after a local capture
```

Runtime auto-loads `skill-task-v1` + `task-v1` when present (override with `NFCM_CODEBOOK` / `NFCM_HYPERNET_CHECKPOINT`).
