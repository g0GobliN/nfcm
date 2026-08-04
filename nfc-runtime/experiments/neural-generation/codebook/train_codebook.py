#!/usr/bin/env python3
"""Train toy skill residuals for the Phase 3 codebook.

Learns 64-d unit vectors per skill with a margin contrastive loss:
  same-domain pairs → cosine near +1
  cross-domain pairs → cosine below a margin

Writes checkpoints/skill-trained-v1.json (NFCM SkillCodebook JSON).

Usage:
  python3 train_codebook.py

No PyTorch required (numpy only). Honest scope: toy fit — not a foundation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIM = 64
STEPS = 2500
LR = 0.05
SEED = 11
MARGIN = 0.2  # want neg cosine < margin

BUNDLES: dict[str, list[str]] = {
    "coding": [
        "python",
        "rust",
        "javascript",
        "typescript",
        "debugging",
        "testing",
        "refactoring",
        "algorithms",
    ],
    "math": ["math", "calculus", "statistics"],
    "writing": ["writing", "summarization"],
    "research": ["research"],
    "medical": ["medical"],
}


def all_skills() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for skills in BUNDLES.values():
        for s in skills:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def main() -> None:
    rng = np.random.default_rng(SEED)
    skills = all_skills()
    idx = {s: i for i, s in enumerate(skills)}
    # Orthogonal-ish init so domains start separable
    vecs = rng.normal(0, 1.0, (len(skills), DIM)).astype(np.float32)
    for i in range(len(skills)):
        vecs[i] = l2_normalize(vecs[i])

    domains = list(BUNDLES.keys())
    skill_domain = {}
    for d, ss in BUNDLES.items():
        for s in ss:
            skill_domain[s] = d

    losses: list[float] = []
    pos_hist: list[float] = []
    neg_hist: list[float] = []

    for step in range(STEPS):
        d_pos = domains[step % len(domains)]
        pos_skills = BUNDLES[d_pos]
        if len(pos_skills) >= 2:
            a_s, b_s = rng.choice(pos_skills, size=2, replace=False)
        else:
            a_s = b_s = pos_skills[0]

        # Hard-ish negative: any skill from a different domain
        candidates = [s for s in skills if skill_domain[s] != d_pos]
        n_s = rng.choice(candidates)

        ia, ib, inn = idx[a_s], idx[b_s], idx[n_s]
        va, vb, vn = vecs[ia], vecs[ib], vecs[inn]

        c_pos = float(np.dot(va, vb))
        c_neg = float(np.dot(va, vn))

        # L_pos: (1 - cos)^2 ; L_neg: max(0, cos - margin)^2
        loss_pos = (1.0 - c_pos) ** 2
        loss_neg = max(0.0, c_neg - MARGIN) ** 2
        loss = loss_pos + loss_neg
        losses.append(loss)
        pos_hist.append(c_pos)
        neg_hist.append(c_neg)

        # Analytic grads on unit sphere (ignore renorm in backward — re-normalize after)
        # d(cos)/dva = vb, etc.
        g_a = np.zeros(DIM, dtype=np.float32)
        g_b = np.zeros(DIM, dtype=np.float32)
        g_n = np.zeros(DIM, dtype=np.float32)

        g_a += -2.0 * (1.0 - c_pos) * vb
        g_b += -2.0 * (1.0 - c_pos) * va
        if c_neg > MARGIN:
            g_a += 2.0 * (c_neg - MARGIN) * vn
            g_n += 2.0 * (c_neg - MARGIN) * va

        vecs[ia] = l2_normalize(va - LR * g_a)
        vecs[ib] = l2_normalize(vb - LR * g_b)
        vecs[inn] = l2_normalize(vn - LR * g_n)

        if step % 500 == 0 or step == STEPS - 1:
            print(
                f"step={step:4d} loss={loss:.4f} "
                f"pos={c_pos:.3f} neg={c_neg:.3f} "
                f"avg_pos={np.mean(pos_hist[-100:]):.3f} avg_neg={np.mean(neg_hist[-100:]):.3f}"
            )

    # Report mean within/across domain cosine
    within, across = [], []
    for i, s1 in enumerate(skills):
        for j, s2 in enumerate(skills):
            if i >= j:
                continue
            c = float(np.dot(vecs[i], vecs[j]))
            if skill_domain[s1] == skill_domain[s2]:
                within.append(c)
            else:
                across.append(c)
    print(
        f"mean_within={np.mean(within):.3f} mean_across={np.mean(across):.3f} "
        f"(want within > across)"
    )

    entries = {s: [round(float(x), 6) for x in vecs[idx[s]]] for s in skills}
    out = {
        "id": "skill-trained-v1",
        "dim": DIM,
        "version": 1,
        "notes": (
            "Toy margin-contrastive skill residuals (numpy, unit vectors). "
            "Same-domain pulled together; cross-domain below margin. Not a foundation model."
        ),
        "entries": entries,
        "train": {
            "steps": STEPS,
            "lr": LR,
            "seed": SEED,
            "margin": MARGIN,
            "final_loss": losses[-1] if losses else None,
            "mean_loss_last_100": float(np.mean(losses[-100:])) if losses else None,
            "mean_within_cosine": float(np.mean(within)) if within else None,
            "mean_across_cosine": float(np.mean(across)) if across else None,
        },
    }

    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "skill-trained-v1.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} ({len(skills)} skills)")


if __name__ == "__main__":
    main()
