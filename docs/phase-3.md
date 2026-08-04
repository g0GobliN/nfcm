# Phase 3 — Skill codebook (compressed knowledge store)

## Goal

Start the **compressed knowledge** seam that NFCM is about:

**store skill residuals once → activate only the slices a task needs → blend into the latent → generate a specialist.**

This is still a **deterministic / untrained** bank. It proves the plumbing, not quality.

## How it works

```text
TaskProfile.skills  →  SkillCodebook.activate()
                     →  residual vector (dim=64)
                     →  added into LatentCode
                     →  topology scales with skill hit count (dynamic specialist)
                     →  LatentWeightGenerator synthesizes tiny tensors
```

- Default bank: builtin + `experiments/neural-generation/codebook/skill-v1.json`
- Trained toy bank (if present): `codebook/checkpoints/skill-trained-v1.json`
- Override: `NFCM_CODEBOOK=/path/to/codebook.json`
- Activated skills are recorded in `LatentCode.codebook_id` (e.g. `skill-codebook-v1+python,debugging`)
- Hot working set is LRU-paged under `NFCM_CODEBOOK_RAM_BYTES` (default: 4 skills). Optimize Memory in the desktop app evicts cold residuals.

## Train a toy codebook

```bash
cd nfc-runtime/experiments/neural-generation/codebook
python3 train_codebook.py
```

## Try it

```bash
cd nfc-runtime
NFCM_WEIGHT_GENERATOR=latent cargo run -p nfc-runtime --example smoke
NFCM_WEIGHT_GENERATOR=latent cargo run -p nfc-runtime --example eval_generators
```

Compile a coding brain in the desktop app (latent generator). Check the loaded brain’s latent `codebook_id` for activated skills.

## Swap in trained residuals later

Train (or export) vectors into the same JSON shape:

```json
{
  "id": "my-trained-v1",
  "dim": 64,
  "version": 1,
  "notes": "trained on …",
  "entries": {
    "python": [ /* 64 floats */ ],
    "debugging": [ /* 64 floats */ ]
  }
}
```

```bash
export NFCM_CODEBOOK=/path/to/my-trained-v1.json
```

No runtime API change required.

## Next

1. Real task-loss residuals (beyond contrastive toy)
2. ~~Page / unload cold skills under RAM pressure~~ — `SkillPager` + `NFCM_CODEBOOK_RAM_BYTES`
3. Score specialists at fixed RAM — see [eval.md](eval.md)
